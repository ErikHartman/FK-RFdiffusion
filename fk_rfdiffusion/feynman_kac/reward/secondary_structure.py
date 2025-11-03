from typing import Tuple
import torch
import pydssp
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def secondary_structure_reward(
    pose,  # Individual pose passed by evaluator
    sequence: str,  # Individual sequence passed by evaluator
    design_chain: str,
    target_alpha: float,
    target_beta: float, 
    target_loop: float,
    weight_alpha: float = 1.0,
    weight_beta: float = 1.0,
    weight_loop: float = 1.0,
    mask: float = 0.0,
    **kwargs
) -> Tuple[float, str, dict]:

    dssp_dict = _compute_secondary_structure_dssp(pose, design_chain, mask)
    ss_seq_dict = _compute_secondary_structure_sequence(sequence)
    ss_dict = {
        'beta_sheet_fraction': (dssp_dict['beta_sheet_fraction']*.8 + ss_seq_dict['beta_sheet_fraction']*.2) ,
        'helix_fraction': (dssp_dict["helix_fraction"]*.8 + ss_seq_dict["helix_fraction"]*.2) ,
        'loop_fraction': (dssp_dict['loop_fraction']*.8 + ss_seq_dict['loop_fraction']*.2)   # Combine DSSP and sequence loop fractions
    }
    reward_value = _get_ss_reward(ss_dict, target_alpha, target_beta, target_loop, 
                                  weight_alpha, weight_beta, weight_loop)
    
    # Print detailed secondary structure information
    print(f"    a={ss_dict['helix_fraction']:.3f} "
          f"B={ss_dict['beta_sheet_fraction']:.3f} "
          f"loop={ss_dict['loop_fraction']:.3f} "
          f"reward={reward_value:.3f}")

    reward_dict = {
        **ss_dict,
        'target_alpha': target_alpha,
        'target_beta': target_beta,
        'target_loop': target_loop,
        'weight_alpha': weight_alpha,
        'weight_beta': weight_beta,
        'weight_loop': weight_loop,
        'reward_value': reward_value,
    }
          
    return reward_value, sequence, reward_dict

def _get_ss_reward(ss_dict: dict, target_alpha: float, target_beta: float, target_loop: float,
                   weight_alpha: float, weight_beta: float, weight_loop: float) -> float:
    alpha_dev = abs(ss_dict['helix_fraction'] - target_alpha)
    beta_dev = abs(ss_dict['beta_sheet_fraction'] - target_beta)
    loop_dev = abs(ss_dict['loop_fraction'] - target_loop)
    
    alpha_reward = weight_alpha * (1.0 - alpha_dev)
    beta_reward = weight_beta * (1.0 - beta_dev)
    loop_reward = weight_loop * (1.0 - loop_dev)
    
    return alpha_reward + beta_reward + loop_reward

def _compute_secondary_structure_dssp(pose, design_chain: str, mask:float) -> dict:
    pdb_info = pose.pdb_info()
    chain_indices = [
        i for i in range(1, pose.total_residue() + 1) 
        if pdb_info.chain(i) == design_chain
    ]
    
    coords = []
    for resi in chain_indices:
        residue = pose.residue(resi)
        n_xyz = residue.xyz("N")
        ca_xyz = residue.xyz("CA") 
        c_xyz = residue.xyz("C")
        o_xyz = residue.xyz("O")
        
        coords.append([
            [n_xyz.x, n_xyz.y, n_xyz.z],
            [ca_xyz.x, ca_xyz.y, ca_xyz.z],
            [c_xyz.x, c_xyz.y, c_xyz.z],
            [o_xyz.x, o_xyz.y, o_xyz.z]
        ])
    
    coord_tensor = torch.tensor(coords, dtype=torch.float32)
    coord_tensor = coord_tensor.unsqueeze(0)
    dssp_result = pydssp.assign(coord_tensor, out_type='index')
    dssp_indices = dssp_result[0]

    if mask > 0.0: # Apply random mask to DSSP indicies
        dssp_indices[torch.rand(len(dssp_indices)) < mask] = -1  # Masked residue

    beta_count = (dssp_indices == 2).sum().item()
    helix_count = (dssp_indices == 1).sum().item()
    loop_count = (dssp_indices == 0).sum().item()
    total_residues = len(dssp_indices)
    
    return {
        'beta_sheet_fraction': beta_count / max(total_residues, 1),
        'helix_fraction': helix_count / max(total_residues, 1),
        'loop_fraction': loop_count / max(total_residues, 1),
    }

def _compute_secondary_structure_sequence(designed_seq: str) -> dict:
    X = ProteinAnalysis(designed_seq)
    sec_struc = X.secondary_structure_fraction() # [helix, turn, sheet]

    return {
        'beta_sheet_fraction': sec_struc[2],
        'helix_fraction': sec_struc[0],
        'loop_fraction': sec_struc[1],
    }