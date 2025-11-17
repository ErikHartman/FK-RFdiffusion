import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd

rfd_path = Path(__file__).parent.parent / "externals" / "RFdiffusion"
sys.path.insert(0, str(rfd_path))

from omegaconf import OmegaConf
from rfdiffusion.inference import utils as iu

from .feynman_kac.feynman_kac import FeynmanKacSampler
from .feynman_kac.reward import ensure_pyrosetta_initialized, get_reward_function
from .utils import get_checkpoint_path, parse_structure_file, cleanup_temp_pdb, infer_design_mode_from_contigs, detect_length_ranges, sample_contig_lengths, validate_symmetry_contigs
from .config.utils import (
    load_config_with_defaults, 
    auto_detect_chain_assignments, 
    validate_config
)


def run_feynman_kac_design(
    contigs: List[str],
    target_structure: Optional[str] = None,
    hotspot_res: Optional[List[str]] = None,
    num_designs: int = 1,
    output_prefix: str = "./design",
    n_particles: int = 10,
    n_runs: int = 1,
    resampling_frequency: int = 5,
    guidance_start_timestep: int = 50,
    num_diffusion_timesteps: int = 50,
    save_full_trajectory: bool = False,
    max_workers: int = 1,
    potential_mode: str = "difference",
    tau: float = None,
    final_step: int = 1,
    checkpoint: str = "base",
    reward_function: Optional[str] = None,
    custom_reward_fn: Optional[callable] = None,
    n_sequences: int = 1,
    aggregation_mode: str = "mean",
    symmetry: Optional[str] = None,
    **kwargs
) -> None:
    """
    Run Feynman-Kac guided protein design.
    
    Args:
        contigs: List of contig specifications. For binder: ['A1-50/0 10'] = target A1-50, chainbreak, 10-residue binder. For unconditional: ['50'] = 50-residue protein
        target_structure: Path to target protein PDB file (required for binder mode, forbidden for unconditional mode)
        design_mode: Design mode - "binder" for protein-protein interactions or "unconditional" for standalone protein design. If None, automatically inferred from contigs
        hotspot_res: List of hotspot residue specifiers (e.g., ["A1","A2","A3"]) - only used in binder mode
        num_designs: Number of designs to generate
        output_prefix: Output file prefix
        n_particles: Number of particles for FK sampling
        n_runs: Number of independent runs to perform (default: 1)
        resampling_frequency: How often to guide + resample particles (default: 5)
        guidance_start_timestep: When to start applying guidance (default: 50). Must be <= num_diffusion_timesteps
        num_diffusion_timesteps: Total number of diffusion timesteps (T parameter, default: 50). 
            Controls how many denoising steps are taken. Higher values = more gradual denoising but slower.
            Typical values: 25-100. Must be >= guidance_start_timestep.
        save_full_trajectory: Whether to save PDB files at every timestep (default: False)
        max_workers: Maximum number of parallel workers (default: min(n_particles, 4))
        potential_mode: Potential function mode. Options:
            - "immediate": G_t = exp(λ * r_φ(x_t))
            - "difference": G_t = exp(λ * (r_φ(x_t) - r_φ(x_{t+1})))  [default]
            - "max": G_t = exp(λ * max_{s=t}^T r_φ(x_s))
            - "sum": G_t = exp(λ * Σ_{s=t}^T r_φ(x_s))
            - "blind": Uniform resampling (for testing)
        tau: FK particle selection tau (1/lambda parameter). If None, uses reward function's recommended value
        final_step: Final diffusion step
        checkpoint: RFdiffusion checkpoint to use - "base" or "beta" (default: "base")
        reward_function: Name of predefined reward function (e.g., "interface_dG", "alpha_helix_ss").
        custom_reward_fn: Custom reward function with signature (pdb_path: str) -> Tuple[float, str, dict].
            Cannot be used together with reward_function.
        n_sequences: Number of sequences to generate per MPNN evaluation for better reward estimates (default: 1)
        aggregation_mode: How to aggregate multiple sequence rewards - "mean" or "max" (default: "mean")
        symmetry: Symmetry specification for symmetric assemblies (e.g., 'C5', 'C3', 'D2', 'tetrahedral', 'icosahedral').
            Contig length must be divisible by symmetry order. Default: None (no symmetry)
        **kwargs: Additional config overrides
        
    Examples:
        >>> run_feynman_kac_design(
        ...     contigs=["A1-50/0 20"],
        ...     target_structure="target.pdb",
        ...     reward_function="interface_dG"
        ... )
        
        Length range sampling (binder with variable length):
        >>> run_feynman_kac_design(
        ...     contigs=["A1-50/0 15-25"],
        ...     target_structure="target.pdb", 
        ...     reward_function="interface_dG",
        ...     n_runs=10
        ... )
        
        Symmetric assembly design (C5 pentamer):
        >>> run_feynman_kac_design(
        ...     contigs=["100"],
        ...     symmetry="C5",
        ...     reward_function="some_reward",
        ...     n_runs=5
        ... )
        
        Unconditional design with length range:
        >>> run_feynman_kac_design(
        ...     contigs=["50-75"],
        ...     reward_function="sequence_hydrophobic",
        ...     n_runs=5
        ... )

    """
    ensure_pyrosetta_initialized()
    design_mode = infer_design_mode_from_contigs(contigs)
    print(f"Design mode: {design_mode}")
    
    # Validate timestep parameters
    if guidance_start_timestep > num_diffusion_timesteps:
        raise ValueError(f"guidance_start_timestep ({guidance_start_timestep}) cannot be greater than "
                        f"num_diffusion_timesteps ({num_diffusion_timesteps})")
    
    # Validate that only one reward method is specified
    if custom_reward_fn is not None and reward_function is not None:
        raise ValueError("Cannot specify both custom_reward_fn and reward_function. Choose one.")
    
    # Validate symmetry and contig compatibility
    if symmetry is not None:
        validate_symmetry_contigs(contigs, symmetry)
        print(f"Symmetry: {symmetry}")
    
    conf = load_config_with_defaults()
    checkpoint_path = get_checkpoint_path(design_mode, checkpoint)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    original_target_structure = target_structure
    temp_pdb_path = None
    
    if target_structure and not Path(target_structure).is_absolute():
        pepdiff_root = Path(__file__).parent.parent
        target_structure_path = Path(target_structure)
        if (Path.cwd() / target_structure_path).exists():
            target_structure = str(Path.cwd() / target_structure_path)
        else:
            target_structure = str(pepdiff_root / target_structure_path)
    
    if target_structure:
        target_structure = parse_structure_file(target_structure)
        if target_structure != original_target_structure:
            temp_pdb_path = target_structure
    elif design_mode == "binder":
        raise ValueError("target_structure must be provided for binder design mode.")

    
    # Apply overrides
    overrides = {
        'inference': {},
        'ppi': {},
        'contigmap': {},
        'diffuser': {},
        'feynman_kac': {},
        'reward': {}
    }
    
    # Set checkpoint path and design mode
    overrides['inference']['ckpt_override_path'] = checkpoint_path
    overrides['inference']['design_mode'] = design_mode
    
    if target_structure is not None:
        overrides['inference']['input_pdb'] = target_structure
    
    if hotspot_res is not None:
        overrides['ppi']['hotspot_res'] = hotspot_res
    
    if symmetry is not None:
        overrides['inference']['symmetry'] = symmetry
    
    # Handle length range sampling for contigs
    has_length_ranges = detect_length_ranges(contigs) if contigs is not None else False
    if has_length_ranges:
        print(f"Detected length ranges in contigs: {contigs}")
        print(f"Will sample {n_runs} different lengths for each run")
        sampled_contigs_list = sample_contig_lengths(contigs, n_runs)
        # Print sampled lengths for each run
        for i, run_contigs in enumerate(sampled_contigs_list):
            print(f"  Run {i+1}: {run_contigs}")
    else:
        # No length ranges, use same contigs for all runs
        sampled_contigs_list = [contigs] * n_runs if contigs is not None else [None] * n_runs
        
    # For initial config creation and validation, use the first sampled contigs
    if contigs is not None:
        overrides['contigmap']['contigs'] = sampled_contigs_list[0]
        
    overrides['inference']['num_designs'] = num_designs
    overrides['inference']['output_prefix'] = output_prefix
    overrides['inference']['final_step'] = final_step
    overrides['feynman_kac']['n_particles'] = n_particles
    overrides['feynman_kac']['n_runs'] = n_runs
    overrides['feynman_kac']['resampling_frequency'] = resampling_frequency
    overrides['feynman_kac']['guidance_start_timestep'] = guidance_start_timestep
    overrides['feynman_kac']['save_full_trajectory'] = save_full_trajectory
    overrides['feynman_kac']['parallel_evaluation'] = True if max_workers > 1 else False
    overrides['feynman_kac']['max_workers'] = max_workers
    overrides['feynman_kac']['potential_mode'] = potential_mode
    overrides['diffuser']['T'] = num_diffusion_timesteps
        
    if reward_function is not None:
        overrides['reward']['function'] = reward_function
    elif custom_reward_fn is not None:
        # Document custom reward function name in config
        overrides['reward']['function'] = f"custom:{custom_reward_fn.__name__}"

    # Set multi-sequence evaluation parameters
    overrides['reward']['n_sequences'] = n_sequences
    overrides['reward']['aggregation_mode'] = aggregation_mode

    if tau is not None:
        overrides['feynman_kac']['tau'] = tau

    # Add any additional kwargs
    for key, value in kwargs.items():
        if "." in key:
            section, param = key.split(".", 1)
            if section not in overrides:
                overrides[section] = {}
            overrides[section][param] = value
        else:
            # Assume it's an inference parameter if no section specified
            overrides['inference'][key] = value
    
    override_conf = OmegaConf.create(overrides)
    conf = OmegaConf.merge(conf, override_conf)
    conf = auto_detect_chain_assignments(conf, design_mode)
    
    validate_config(conf)
    run_guided_inference(conf, contigs, sampled_contigs_list, custom_reward_fn)

    if temp_pdb_path and original_target_structure:
        cleanup_temp_pdb(temp_pdb_path, original_target_structure)


def run_guided_inference(conf, original_contigs: List[str], sampled_contigs_list: List[List[str]], custom_reward_fn=None):
    """Run guided inference with Feynman-Kac sampler"""
    
    if custom_reward_fn is not None:
        # Auto-wrap custom reward function if it's not already a MultiSequenceEvaluator
        from .feynman_kac.reward.base import MultiSequenceEvaluator
        from functools import partial
        
        if not isinstance(custom_reward_fn, MultiSequenceEvaluator):
            # Wrap the raw reward function
            mpnn_config = {
                'mpnn_temperature': float(conf.mpnn.mpnn_temperature),
                'batch_size': int(conf.mpnn.batch_size),
                'use_soluble_model': bool(conf.mpnn.use_soluble_model),
                'suppress_print': bool(conf.mpnn.suppress_print),
                'save_score': bool(conf.mpnn.save_score),
                'save_probs': bool(conf.mpnn.save_probs),
                'design_chains': str(conf.mpnn.design_chains) if conf.mpnn.design_chains else None,
                'fixed_chains': str(conf.mpnn.fixed_chains) if conf.mpnn.fixed_chains else None,
            }
            
            configured_reward_fn = MultiSequenceEvaluator(
                single_sequence_evaluator=partial(custom_reward_fn),
                design_chain=str(conf.reward.design_chain),
                mpnn_config=mpnn_config,
                n_sequences=conf.reward.n_sequences,
                aggregation_mode=conf.reward.aggregation_mode,
                is_symmetric=bool(conf.inference.symmetry is not None and conf.inference.symmetry != '')
            )
        else:
            configured_reward_fn = custom_reward_fn
    else:
        configured_reward_fn = get_reward_function(conf)
    print(OmegaConf.to_yaml(conf))
    output_prefix = conf.inference.output_prefix
    output_dir = Path(output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a config with original contigs (including ranges) for saving
    original_conf = conf.copy()
    if original_contigs is not None:
        original_conf.contigmap.contigs = original_contigs
    config_save_path = output_dir / "config.yaml"
    OmegaConf.save(original_conf, config_save_path)

    # Check if output directory already has results
    output_path = Path(output_prefix)
    if output_path.exists():
        if output_path.is_dir() and "trajectories" in [f.name for f in output_path.iterdir()] \
                                    or "results.csv" in [f.name for f in output_path.iterdir()]:
            raise ValueError(f"Output directory '{output_path}' already exists and contains trajectories or results. "
                           f"Please choose a different output path or remove the existing directory.")
    
    n_runs = conf.feynman_kac.n_runs
    all_results = []
    
    for run_id in range(1, n_runs + 1):
        print(f"\n{'='*60}")
        print(f"Starting run {run_id}/{n_runs}")
        print(f"{'='*60}")
        
        # Create run-specific config with sampled contigs (no ranges)
        run_conf = conf.copy()
        if sampled_contigs_list[run_id - 1] is not None:
            run_conf.contigmap.contigs = sampled_contigs_list[run_id - 1]
            print(f"Using contigs for this run: {sampled_contigs_list[run_id - 1]}")
        
        # Create sampler with run-specific config
        # Note: diffuser.T override is handled in the RFdiffusion model_runners.py
        sampler = iu.sampler_selector(run_conf)
        
        # Determine output prefix for this run
        if n_runs > 1:
            run_output_prefix = str(output_path / f"run_{run_id}")
        else:
            run_output_prefix = str(output_path)
        
        fk_sampler = FeynmanKacSampler(
            base_sampler=sampler,
            n_particles=conf.feynman_kac.n_particles,
            output_prefix=run_output_prefix,
            resampling_frequency=conf.feynman_kac.resampling_frequency,
            reward_fn=configured_reward_fn,
            guidance_start_timestep=conf.feynman_kac.guidance_start_timestep,
            save_full_trajectory=conf.feynman_kac.save_full_trajectory,
            parallel_evaluation=conf.feynman_kac.parallel_evaluation,
            max_workers=conf.feynman_kac.max_workers,
            tau=conf.feynman_kac.tau,
            potential_mode=conf.feynman_kac.potential_mode
        )
        
        # Run the FK diffusion and get results dataframe
        run_results_df = fk_sampler.run_feynman_kac_diffusion()
        
        # Add run column
        run_results_df['run'] = run_id
        all_results.append(run_results_df)
        
        # Save intermittent CSV for this run
        intermittent_csv_path = output_path / f"results_run_{run_id}.csv"
        run_results_df.to_csv(intermittent_csv_path, index=False)
        print(f"Saved intermittent results to {intermittent_csv_path}")
        
        print(f"Completed run {run_id}/{n_runs}")
    
    # Concatenate all results and save
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    results_file = output_path / "results.csv"
    combined_results.to_csv(results_file, index=False)
    print(f"\nSaved combined results from {n_runs} runs to {results_file}")
    print(f"Total particles: {len(combined_results)}")
    print(f"Particles per run: {len(combined_results) // n_runs}")
    
    # Delete intermittent run files now that we have the combined results
    intermittent_files = sorted(output_path.glob("results_run_*.csv"))
    if intermittent_files:
        print(f"\nCleaning up {len(intermittent_files)} intermittent CSV files...")
        for intermittent_file in intermittent_files:
            intermittent_file.unlink()
            print(f"Deleted {intermittent_file.name}")

