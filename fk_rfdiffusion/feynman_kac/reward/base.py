import os
import shutil
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List
import numpy as np

from pyrosetta import init, pose_from_pdb
import pyrosetta.rosetta as rosetta
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.core.pack.task import TaskFactory

_pyrosetta_initialized = False

def ensure_pyrosetta_initialized():
    """Initialize PyRosetta once per process to avoid multiprocessing conflicts"""
    global _pyrosetta_initialized
    if not _pyrosetta_initialized:
        init_flags = "-beta_nov16 -in:file:silent_struct_type binary -use_terminal_residues true -mute all basic.io.database core.scoring"
        init(init_flags)
        _pyrosetta_initialized = True

def run_mpnn_and_thread(pdb_path: str, design_chain: str, mpnn_config: dict = None, n_sequences: int = 1) -> Tuple[List[object], List[str]]:
    """
    Run ProteinMPNN and thread multiple sequences.
    Returns lists of (threaded_poses, designed_sequences) for multiple evaluations.
    """
    pdb_path = Path(pdb_path).resolve()
    output_dir = Path(tempfile.mkdtemp(prefix="mpnn_"))
    designed_sequences = _run_proteinmpnn(
        pdb_path, design_chain, output_dir, mpnn_config or {}, n_sequences
    )
    
    threaded_poses = []
    
    for designed_seq in designed_sequences:
        threaded_pose = _thread_sequence_onto_structure(
            pdb_path, designed_seq, design_chain, output_dir
        )
        threaded_poses.append(threaded_pose)
    
    shutil.rmtree(output_dir)
    return threaded_poses, designed_sequences

def pack_and_minimize_pose(pose):
    """Sidechain packing and minimization for structure refinement"""
    sfxn = get_score_function()
    
    # Step 1: Initial minimization (side chains only) 
    movemap = MoveMap()
    movemap.set_bb(False)  # Keep backbone fixed
    movemap.set_chi(True)  # Allow side chain movement
    
    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(sfxn)
    min_mover.tolerance(0.01)  # Tighter convergence for better refinement
    min_mover.apply(pose)
    
    # Step 2: Pack rotamers
    task_factory = TaskFactory()
    task_factory.push_back(rosetta.core.pack.task.operation.InitializeFromCommandline())
    task_factory.push_back(rosetta.core.pack.task.operation.RestrictToRepacking())
    
    pack_mover = PackRotamersMover()
    pack_mover.task_factory(task_factory)
    pack_mover.score_function(sfxn)
    pack_mover.apply(pose)
    
    # Step 3: Final minimization with tighter tolerance
    min_mover.tolerance(0.001)  # Even tighter for final polish
    min_mover.apply(pose)


class MultiSequenceEvaluator:
    """
    Class that converts a single-sequence reward function into a multi-sequence aggregated one.
    This approach avoids pickling issues with nested functions in multiprocessing.
    """
    
    def __init__(self, single_sequence_evaluator, design_chain: str, mpnn_config: dict, 
                 n_sequences: int = 1, aggregation_mode: str = "mean", **kwargs):
        self.single_sequence_evaluator = single_sequence_evaluator
        self.design_chain = design_chain
        self.mpnn_config = mpnn_config or {}
        self.n_sequences = n_sequences
        self.aggregation_mode = aggregation_mode
        self.kwargs = kwargs
    
    def __call__(self, pdb_path: str) -> Tuple[float, str, dict]:
        """
        Evaluate multiple sequences and return aggregated results.
        """
        if self.n_sequences == 1:
            # Single sequence evaluation - call MPNN and thread once
            poses, sequences = run_mpnn_and_thread(pdb_path, self.design_chain, self.mpnn_config, 1)
            pose, sequence = poses[0], sequences[0]
            pack_and_minimize_pose(pose)
            return self.single_sequence_evaluator(pose, sequence, self.design_chain, **self.kwargs)
        
        # Multi-sequence evaluation
        poses, sequences = run_mpnn_and_thread(pdb_path, self.design_chain, self.mpnn_config, self.n_sequences)
        
        rewards = []
        reward_dicts = []
        
        print(f"Evaluating {len(sequences)} MPNN sequences with aggregation_mode='{self.aggregation_mode}'")
        
        for i, (pose, seq) in enumerate(zip(poses, sequences)):
            pack_and_minimize_pose(pose)
            # Call the original single-sequence function
            reward, _, reward_dict = self.single_sequence_evaluator(pose, seq, self.design_chain, **self.kwargs)
            rewards.append(reward)
            reward_dicts.append(reward_dict)
        
        rewards = np.array(rewards)
        
        # Determine final reward based on aggregation mode
        if self.aggregation_mode == "mean":
            final_reward = np.mean(rewards)
            best_idx = np.argmax(rewards)  # Still save best structure
        elif self.aggregation_mode == "max":
            final_reward = np.max(rewards)
            best_idx = np.argmax(rewards)
        else:
            raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}. Use 'mean' or 'max'")
        
        # Enhanced metadata with all statistics
        aggregated_dict = {
            'final_reward': final_reward,
            'aggregation_mode': self.aggregation_mode,
            'mean_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards),
            'n_sequences': len(rewards),
            'best_sequence': sequences[best_idx],
            'all_sequences': sequences,
            'all_rewards': rewards.tolist()
        }
        
        # Add best individual reward dict (preserving original metrics)
        best_reward_dict = reward_dicts[best_idx].copy()
        best_reward_dict.update(aggregated_dict)
        
        # Save the best pose (highest reward)
        best_pose = poses[best_idx]
        final_pdb_path = save_pose(best_pose, pdb_path)
        best_reward_dict['pdb_path'] = final_pdb_path
        
        print(f"Final aggregated reward ({self.aggregation_mode}): {final_reward:.3f}")
        print(f"Best sequence: {sequences[best_idx]}")
        
        return final_reward, sequences[best_idx], best_reward_dict


def save_pose(pose, pdb_path: str) -> str:
    """Save the final refined pose after packing and minimization"""
    pdb_path_obj = Path(pdb_path).resolve()
    refined_pdb = pdb_path_obj.parent / f"{pdb_path_obj.stem}_refined.pdb"
    pose.dump_pdb(str(refined_pdb))
    return str(refined_pdb)

def _run_proteinmpnn(
    pdb_path: Path, 
    design_chain: str, 
    output_dir: Path,
    mpnn_config: dict,
    n_sequences: int
) -> List[str]:
    """Run ProteinMPNN and extract all designed sequences"""
    
    # MPNN configuration - these should all be present or it's a config error
    mpnn_temperature = mpnn_config["mpnn_temperature"]
    mpnn_batch_size = mpnn_config["batch_size"]
    mpnn_use_soluble = mpnn_config["use_soluble_model"]
    mpnn_save_score = mpnn_config["save_score"]
    mpnn_save_probs = mpnn_config["save_probs"]

    mpnn_path = Path(__file__).parent.parent.parent.parent / "externals" / "ProteinMPNN"

    mpnn_script = mpnn_path / "protein_mpnn_run.py"
    mpnn_out = output_dir / "mpnn_output"
    mpnn_out.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable,
        str(mpnn_script),
        "--pdb_path", str(pdb_path),
        "--pdb_path_chains", design_chain,
        "--out_folder", str(mpnn_out),
        "--num_seq_per_target", str(n_sequences),
        "--sampling_temp", str(mpnn_temperature),
        "--batch_size", str(mpnn_batch_size),
    ]
    
    if mpnn_use_soluble:
        cmd.append("--use_soluble_model")
        
    if mpnn_save_score:
        cmd.append("--save_score")
    if mpnn_save_probs:
        cmd.append("--save_probs")
    
    result = subprocess.run(
        cmd,
        cwd=str(mpnn_path),
        capture_output=True,
        text=True,
        env=os.environ.copy()
    )
    
    # Check if ProteinMPNN command succeeded
    if result.returncode != 0:
        print(f"ProteinMPNN failed with return code {result.returncode}")
        print(f"Command: {' '.join(cmd)}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"Working directory: {mpnn_path}")
        print(f"Output directory: {mpnn_out}")
        raise RuntimeError(f"ProteinMPNN execution failed: {result.stderr}")
    
    seq_files = list((mpnn_out / "seqs").glob("*.fa"))
    
    # Check if any sequence files were generated
    if not seq_files:
        print(f"No sequence files found in {mpnn_out / 'seqs'}")
        print(f"ProteinMPNN command: {' '.join(cmd)}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"Contents of output directory:")
        
        for item in mpnn_out.rglob("*"):
            print(f"  {item}")
        raise RuntimeError("ProteinMPNN did not generate any sequence files")

    with open(seq_files[0], 'r') as f:
        lines = f.readlines()
        sequences = []
        for i, line in enumerate(lines):
            if line.startswith('>'):
                if i + 1 < len(lines):
                    sequence = lines[i + 1].strip()
                    sequences.append(sequence)
        
        # Filter out invalid sequences and return all valid ones
        valid_sequences = []
        for seq in sequences:
            if 'X' not in seq and len(seq) > 0:
                valid_sequences.append(seq)
        
        if not valid_sequences:
            raise RuntimeError("No valid sequences generated by ProteinMPNN (all contain 'X' or are empty)")
        
        return valid_sequences

def _thread_sequence_onto_structure(
    pdb_path: Path,
    designed_seq: str,
    design_chain: str,
    output_dir: Path,
):
    """Thread designed sequence onto structure, return pose"""
    
    cleaned_pdb = output_dir / "cleaned_structure.pdb"
    _clean_rfdiffusion_pdb(pdb_path, cleaned_pdb)
    
    pose = pose_from_pdb(str(cleaned_pdb))
    pdb_info = pose.pdb_info()
    
    # Find residues in design chain and thread sequence
    chain_indices = [
        i for i in range(1, pose.total_residue() + 1) 
        if pdb_info.chain(i) == design_chain
    ]
    
    rsd_set = pose.residue_type_set_for_pose(rosetta.core.chemical.FULL_ATOM_t)
    
    for offset, aa in enumerate(designed_seq):
        resi = chain_indices[offset]
        aa_enum = rosetta.core.chemical.aa_from_oneletter_code(aa)
        name3 = rosetta.core.chemical.name_from_aa(aa_enum)
        new_res = rosetta.core.conformation.ResidueFactory.create_residue(
            rsd_set.name_map(name3)
        )
        pose.replace_residue(resi, new_res, True)

    return pose

def _clean_rfdiffusion_pdb(pdb_path: Path, output_path: Path) -> Path:
    """Clean RFdiffusion PDB by replacing MAS with ALA"""
    with open(pdb_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') and 'MAS' in line:
                line = line[:17] + 'ALA' + line[20:]
            f_out.write(line)
    return output_path

