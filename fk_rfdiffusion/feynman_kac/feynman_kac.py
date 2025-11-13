import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from rfdiffusion.util import writepdb

rfd_path = Path(__file__).parent.parent.parent / "externals" / "RFdiffusion"
if str(rfd_path) not in sys.path:
    sys.path.insert(0, str(rfd_path))



class FeynmanKacSampler:
    """
    Feynman-Kac particle filtering for guided diffusion.

    Diffusion goes from t=T (configurable via diffuser.T, default 50) to t=1, 
    with guidance starting at guidance_start_timestep.
    """
    
    def __init__(self, 
                 base_sampler,
                 n_particles: int = 10,
                 output_prefix: str = "./design",
                 resampling_frequency: int = 5,
                 reward_fn = None,
                 guidance_start_timestep: int = 50,
                 save_full_trajectory: bool = False,
                 parallel_evaluation: bool = True,
                 max_workers: int = None,
                 tau: float = 10.0,
                 potential_mode: str = "immediate"):

        self.base_sampler = base_sampler
        self.n_particles = n_particles
        self.output_prefix = output_prefix
        self.resampling_frequency = resampling_frequency
        self.reward_fn = reward_fn
        self.guidance_start_timestep = guidance_start_timestep
        self.save_full_trajectory = save_full_trajectory
        self.parallel_evaluation = parallel_evaluation 
        self.max_workers = max_workers or min(n_particles, 4)
        self.tau = tau
        self.potential_mode = potential_mode
    
        
        self.particle_reward_history = {}  # {particle_name: [reward_t50, reward_t45, ...]}

        self.metadata_records = []  # List of dicts for CSV output
        self.particle_counter = 0  # Global counter for unique particle names

    def generate_unique_particle_name(self) -> str:
        """Generate a unique particle name that persists across iterations"""
        name = f"p{self.particle_counter:04d}"
        self.particle_counter += 1
        return name

    def take_step(self, x_t, seq_t, t: int, final_step: int, particle_idx: int, particle_name: str,  new_particles_x: List, 
                  new_particles_seq: List, all_px0: List) -> bool:
        """
        Take a diffusion step for a single particle with robust error handling.
        
        Returns True if successful, False if failed.
        Modifies new_particles_x, new_particles_seq, and all_px0 in place.
        """
        print(f"Taking step for particle {particle_idx} ({particle_name}) at t={t}")
        max_retries = 5
        retry_count = 0
        original_x_t = x_t.clone()  # Keep original coordinates for jittering
        
        while retry_count < max_retries:
            try:
                # Add small jitter on retries to break numerical instabilities
                if retry_count > 0:
                    jitter_scale = 1e-4 * retry_count  # Increase jitter with retry count
                    jitter = torch.randn_like(original_x_t[:, :3]) * jitter_scale  # Only jitter backbone atoms
                    x_t_jittered = original_x_t.clone()
                    x_t_jittered[:, :3] += jitter
                    current_x_t = x_t_jittered
                else:
                    current_x_t = x_t
                
                px0, x_next, seq_next, _ = self.base_sampler.sample_step( 
                    t=t, x_t=current_x_t, seq_init=seq_t, final_step=final_step
                ) # Default behavior with 0 retries
                
                # Check for NaNs in backbone atoms (N, CA, C - first 3 atoms)
                if torch.isnan(x_next[:, :3]).any() or torch.isnan(px0[:, :3]).any():
                    retry_count += 1
                    print(f"Particle {particle_idx} ({particle_name}) produced NaN coordinates at t={t}, retrying ({retry_count}/{max_retries})...")
                    continue  # Retry with jitter
                
                new_particles_x.append(x_next.clone())
                new_particles_seq.append(seq_next.clone())
                all_px0.append(px0.clone())
                
                if retry_count > 0:
                    print(f"Particle {particle_idx} ({particle_name}) succeeded after {retry_count + 1} attempts at t={t}")
                return True
                
            except (np.linalg.LinAlgError, ValueError) as e:
                if "SVD did not converge" in str(e) or "rotation matrix" in str(e).lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Particle {particle_idx} ({particle_name}) failed attempt {retry_count} at t={t} with {str(e)}, retrying with jitter...")
                    else:
                        print(f"Particle {particle_idx} ({particle_name}) failed all {max_retries} attempts at t={t}")
                else:
                    raise e
        
        raise RuntimeError(f"Particle {particle_idx} ({particle_name}) failed all {max_retries} attempts at t={t} (both SVD errors and NaN coordinates)")

    def evaluate_particles_parallel(self, px0_pdb_paths: List[str]) -> Tuple[List[float], List[str], List[dict]]:
        """
        Evaluate multiple particles in parallel using ProcessPoolExecutor
        """
        print(f"{len(px0_pdb_paths)} particles to evaluate")
        if not self.parallel_evaluation:
            rewards = []
            sequences = []
            reward_dicts = []
            for pdb_path in px0_pdb_paths:
                reward, seq, reward_dict = self.reward_fn(pdb_path)
                rewards.append(reward)
                sequences.append(seq)
                reward_dicts.append(reward_dict)
            return rewards, sequences, reward_dicts
        
        rewards = [None] * len(px0_pdb_paths)
        sequences = [None] * len(px0_pdb_paths)
        reward_dicts = [None] * len(px0_pdb_paths)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.reward_fn, pdb_path): i 
                for i, pdb_path in enumerate(px0_pdb_paths)
            }
            print("Waiting for particle evaluations to complete...")
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                reward, seq, reward_dict = future.result()
                rewards[i] = reward
                sequences[i] = seq
                reward_dicts[i] = reward_dict

        return rewards, sequences, reward_dicts
    
    def select_particles(self, current_rewards: List[float], particle_names: List[str], sequences: List[str], 
                        current_timestep: int = None, n_particles: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute potential values and select particles based on them.
        Returns both selected indices and potential values for metadata logging.
        """
        current_rewards = np.array(current_rewards)
        
        # Compute f(r) values based on potential mode
        if self.potential_mode == "immediate":
            # G_t = exp(r_φ(x_t) / tau)
            f_values = current_rewards
        
        elif self.potential_mode == "difference":
            # G_t = exp((r_φ(x_t) - r_φ(x_{t+1})) / tau)
            # G_T = 1 (uniform at first guidance step)
            if current_timestep == self.guidance_start_timestep:
                # First guidance step: uniform potentials
                f_values = np.zeros(len(current_rewards))  # exp(0) = 1
            else:
                f_values = []
                for i, particle_name in enumerate(particle_names):
                    hist = self.particle_reward_history.get(particle_name, [])
                    # History should always have >= 2 elements here (previous + current)
                    previous_reward = hist[-2]  # previous guidance step
                    difference = current_rewards[i] - previous_reward
                    f_values.append(difference)
                f_values = np.array(f_values)
        
        elif self.potential_mode == "max":
            # G_t = exp(max_{s=t}^T r_φ(x_s) / tau)
            f_values = []
            for i, particle_name in enumerate(particle_names):
                if particle_name not in self.particle_reward_history or len(self.particle_reward_history[particle_name]) == 0:
                    # Should only happen on very first guidance step
                    max_reward = current_rewards[i]
                else:
                    # Max of all rewards in history (current reward already included)
                    max_reward = max(self.particle_reward_history[particle_name])
                f_values.append(max_reward)
            f_values = np.array(f_values)
        
        elif self.potential_mode == "sum":
            # G_t = exp(Σ_{s=t}^T r_φ(x_s) / tau)
            if current_timestep == self.guidance_start_timestep:
                f_values = current_rewards
            else:
                f_values = []
                for i, particle_name in enumerate(particle_names):
                    # Sum of all rewards in history (current reward already included)
                    # Normalize by number of rewards to keep scale reasonable for tau
                    sum_reward = sum(self.particle_reward_history[particle_name]) / len(self.particle_reward_history[particle_name])
                    f_values.append(sum_reward)
                f_values = np.array(f_values)
            
        elif self.potential_mode == "blind":
            # G_t = 1 (uniform)
            f_values = np.zeros(len(current_rewards))  # exp(0) = 1

        # Compute potential values G_t = exp(f(r) / tau) with numerical stability
        x = f_values / self.tau
        x = x - np.max(x)  # Stable computation: subtract max to prevent overflow
        x = np.clip(x, -1e3, None)  # Clip to prevent underflow
        potential_values = np.exp(x)  # Now safe to exponentiate: G_t = exp(f(r) / tau)
        
        # Handle blind mode or uniform selection
        if self.potential_mode == "blind": 
            selected_indices = list(range(n_particles))
            print(f"Blind mode. Selected indices: {selected_indices}")
            return np.array(selected_indices), potential_values
        
        # Particle selection based on potential values
        rng = np.random.default_rng()
        potentials = np.asarray(potential_values, dtype=np.float64)
        rewards_array = np.asarray(current_rewards, dtype=np.float64)
        
        bad = ~np.isfinite(potentials)
        idx = np.where(~bad)[0]
        rewards_ok = rewards_array[idx]
        potentials_ok = potentials[idx]
        f_values_ok = f_values[idx]
        sequences_ok = [sequences[i] for i in idx]
        Z = potentials_ok.sum()

        if not np.isfinite(Z) or Z <= 0.0:
            p = np.full_like(potentials_ok, 1.0 / potentials_ok.size)
            print("Normalization failed - using uniform probabilities")
        else:
            p = potentials_ok / Z
            p = p / p.sum()

        for i, (particle_idx, prob) in enumerate(zip(idx, p)):
            print(f"  Particle {particle_idx}: Reward={rewards_ok[i]:.3f}, f(r)={f_values_ok[i]:.3f}, Potential={potentials_ok[i]:.3f}, prob={prob:.3f}, seq={sequences_ok[i][:32]}...")
        
        choice_local = rng.choice(len(idx), size=n_particles, replace=True, p=p)
        selected_indices = idx[choice_local]

        ESS = 1.0 / np.sum(p**2)
        print(f"Effective Sample Size (ESS): {ESS:.2f} out of {len(rewards_ok)} successful particles")
        print(f"Selected particle indices: {selected_indices}")
        return selected_indices, potential_values

    def update_reward_history(self, particle_names: List[str], rewards: List[float]):
        for particle_name, reward in zip(particle_names, rewards):
            if particle_name not in self.particle_reward_history:
                self.particle_reward_history[particle_name] = []
            self.particle_reward_history[particle_name].append(reward)
    
    def run_feynman_kac_diffusion(self):
        start_time_dt = datetime.datetime.now()
        start_time = start_time_dt.strftime("%Y%m%d_%H%M%S")
        print(f"Starting step-by-step Feynman-Kac sampling with {self.n_particles} particles at {start_time}")

        # Initialize particles at starting timestep
        final_step = 1
        start_timestep = int(self.base_sampler.t_step_input)
        
        # Particle state: list of (x_t, seq_t) tuples
        particles_x = []
        particles_seq = []
        particle_unique_names = []
        particle_parents = []
        
        # Initialize N particles
        for i in range(self.n_particles):
            x_init, seq_init = self.base_sampler.sample_init()
            particles_x.append(x_init.clone())
            particles_seq.append(seq_init.clone())
            particle_unique_names.append(self.generate_unique_particle_name())
            particle_parents.append(None)
        
        # Step-by-step diffusion with guidance
        for t in range(start_timestep, final_step - 1, -1):
            print(f"Timestep {t}: Processing {len(particles_x)} particles")
            
            is_guidance_step = (t <= self.guidance_start_timestep and 
                              t % self.resampling_frequency == 0 and 
                              t > final_step)
            
            # Save x_t trajectories BEFORE stepping (save current state at timestep t)
            if (self.save_full_trajectory or is_guidance_step) and t > final_step:
                for i in range(len(particles_x)):
                    self.save_particle_structure(
                        particles_x[i], particles_seq[i], t, particle_unique_names[i]
                    )
            
            # Step 1: Take single diffusion step for all particles
            new_particles_x = []
            new_particles_seq = []
            all_px0 = []
            
            for i, (x_t, seq_t) in enumerate(zip(particles_x, particles_seq)):
                self.take_step(x_t, seq_t, t, final_step, i, particle_unique_names[i], new_particles_x, new_particles_seq, all_px0)
            
            # Step 2: Guidance + Resampling - if this is a resampling timestep
            if is_guidance_step:
                print(f"Guidance step at timestep {t} (potential_mode={self.potential_mode})")
                
                # Save all px0 structures and collect paths for parallel evaluation. This is px0 for timestep t.
                px0_pdb_paths = []
                for i, px0 in enumerate(all_px0):
                    px0_pdb_path = self.save_particle_structure(
                        px0, particles_seq[i], t, particle_unique_names[i], is_px0=True
                    )
                    px0_pdb_paths.append(px0_pdb_path)
                
                guidance_rewards, designed_sequences, reward_dicts = self.evaluate_particles_parallel(px0_pdb_paths)
                self.update_reward_history(particle_unique_names, guidance_rewards)
                
                # Select particles using unified function
                selected_indices, potential_values = self.select_particles(
                    current_rewards=guidance_rewards,
                    particle_names=particle_unique_names,
                    sequences=designed_sequences,
                    current_timestep=t,
                    n_particles=self.n_particles
                )
                
                # Record metadata for all particles
                for i, (px0_reward, designed_sequence, reward_dict, px0_pdb_path) in enumerate(
                    zip(guidance_rewards, designed_sequences, reward_dicts, px0_pdb_paths)
                ):
                    metadata_record = {
                        'iteration': t,
                        'particle_name': particle_unique_names[i],
                        'parent_particle': particle_parents[i],
                        'reward': px0_reward.item() if torch.is_tensor(px0_reward) else px0_reward,
                        'potential_value': potential_values[i],
                        'potential_mode': self.potential_mode,
                        'sequence': designed_sequence,
                        'pdb_path': px0_pdb_path
                    }
                    metadata_record.update(reward_dict)
                    self.metadata_records.append(metadata_record)
                
                # Create resampled particle set
                resampled_particles_x = []
                resampled_particles_seq = []
                resampled_names = []
                resampled_parents = []
                
                for selected_idx in selected_indices:
                    resampled_particles_x.append(new_particles_x[selected_idx].clone())
                    resampled_particles_seq.append(new_particles_seq[selected_idx].clone())
                    resampled_parents.append(particle_unique_names[selected_idx])  # Track lineage
                    new_name = self.generate_unique_particle_name()
                    resampled_names.append(new_name)   # New unique name
                    
                    # Copy reward history from parent to child INCLUDING current reward
                    parent_name = particle_unique_names[selected_idx]
                    self.particle_reward_history[new_name] = self.particle_reward_history[parent_name].copy()

                particles_x = resampled_particles_x
                particles_seq = resampled_particles_seq
                particle_unique_names = resampled_names
                particle_parents = resampled_parents
            else:
                particles_x = new_particles_x
                particles_seq = new_particles_seq
        
        # Save all final structures and collect paths for parallel evaluation
        final_pdb_paths = []
        for i, (x_final, seq_final) in enumerate(zip(particles_x, particles_seq)):
            final_pdb_path = self.save_particle_structure(
                x_final, seq_final, final_step, particle_unique_names[i]
            )
            final_pdb_paths.append(final_pdb_path)
        
        final_rewards, final_sequences, final_reward_dicts = self.evaluate_particles_parallel(final_pdb_paths)
        
        # Update reward history for final particles (no resampling needed for final step)
        self.update_reward_history(particle_unique_names, final_rewards)
        for i, (final_reward, designed_sequence, reward_dict, final_pdb_path) in enumerate(
            zip(final_rewards, final_sequences, final_reward_dicts, final_pdb_paths)
        ):
            metadata_record = {
                'iteration': final_step,
                'particle_name': particle_unique_names[i],
                'parent_particle': particle_parents[i],
                'reward': final_reward.item() if torch.is_tensor(final_reward) else final_reward,
                'potential_value': None,  # No resampling needed for final step
                'potential_mode': self.potential_mode,
                'sequence': designed_sequence,
                'pdb_path': final_pdb_path
            }
            metadata_record.update(reward_dict)
            self.metadata_records.append(metadata_record)
        
        df = pd.DataFrame(self.metadata_records)
        
        end_time_dt = datetime.datetime.now()
        end_time = end_time_dt.strftime("%Y%m%d_%H%M%S")
        elapsed_time = (end_time_dt - start_time_dt).total_seconds()
        print(f"Run completed at {end_time}, time taken: {elapsed_time:.1f} seconds")
        
        return df
    
    def save_particle_structure(self, coords, seq, t: int, unique_name: str, is_px0: bool = False) -> str:
        """
        Save PDB file for particle structure at specific timestep
        Unified function for both px0 (denoised) and x_t (noisy) structures
        """
        output_path = Path(self.output_prefix)
        output_path.mkdir(exist_ok=True)
        trajectories_dir = output_path / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        
        timestep_dir = trajectories_dir / f"t_{t:02d}"
        timestep_dir.mkdir(exist_ok=True)
        
        if seq.dim() > 1:
            seq_final = torch.argmax(seq, dim=-1)
        else:
            seq_final = seq

        prefix = "px0_" if is_px0 else ""
        filename = f"{prefix}{unique_name}.pdb"
        pdb_path = timestep_dir / filename
  
        writepdb(
            str(pdb_path),
            coords[:, :4].clone(),
            seq_final,
            self.base_sampler.binderlen,
            chain_idx=getattr(self.base_sampler, 'chain_idx', None),
            bfacts=None,
            idx_pdb=getattr(self.base_sampler, 'idx_pdb', None)
        )
        
        return str(pdb_path)