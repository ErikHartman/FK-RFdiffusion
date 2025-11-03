from typing import Tuple

from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.scoring import get_score_function

def dG_reward(
    pose,  # Individual pose passed by evaluator
    sequence: str,  # Individual sequence passed by evaluator
    design_chain: str,
    target_chain: str,
    **kwargs
) -> Tuple[float, str, dict]:
    
    # Pose is already threaded and packed by decorator
    interface_label = f"{target_chain}_{design_chain}"
    sfxn = get_score_function()
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_label)
    iam.set_scorefunction(sfxn)
    iam.apply(pose)
    final_dG = float(iam.get_interface_dG())
    
    print(f"    ΔG={final_dG:.3f}")
    
    reward_dict = {
        'interface_dG': final_dG,
        'interface_label': interface_label,
        'target_chain': target_chain,
        'design_chain': design_chain
    }
    
    reward_value = -final_dG
    return reward_value, sequence, reward_dict

def dSASA_reward(
    pose,  # Individual pose passed by evaluator
    sequence: str,  # Individual sequence passed by evaluator
    design_chain: str,
    target_chain: str,
    **kwargs
) -> Tuple[float, str, dict]:
    """
    SASA-based reward function.
    """
    interface_label = f"{target_chain}_{design_chain}"
    sfxn = get_score_function()
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_label)
    iam.set_scorefunction(sfxn)
    iam.apply(pose)
    interface_dsasa = float(iam.get_interface_delta_sasa())
    
    # Print detailed SASA information
    print(f"    ΔSASA={interface_dsasa:.1f}")
    
    reward_dict = {
        'interface_dsasa': interface_dsasa,
        'design_chain': design_chain
    }

    reward_value = interface_dsasa
    return reward_value, sequence, reward_dict

def dGdSASA_reward(
    pose,  # Individual pose passed by evaluator
    sequence: str,  # Individual sequence passed by evaluator
    design_chain: str,
    target_chain: str,
    **kwargs
) -> Tuple[float, str, dict]:
    """
    dG/dSASA-based reward function.
    """
    interface_label = f"{target_chain}_{design_chain}"
    sfxn = get_score_function()
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_label)
    iam.set_scorefunction(sfxn)
    iam.apply(pose)
    interface_dsasa = float(iam.get_interface_delta_sasa()) 
    dG = float(iam.get_interface_dG())

    dG_over_sasa = dG / interface_dsasa if interface_dsasa != 0 else 0.0
    
    # Print detailed dG/SASA information
    print(f"    ΔG={dG:.3f} ΔSASA={interface_dsasa:.1f} ΔG/ΔSASA={dG_over_sasa:.4f}")

    reward_dict = {
        'dG_over_sasa': dG_over_sasa,
        'interface_dG': dG,
        'interface_dsasa': interface_dsasa,
        'design_chain': design_chain
    }

    reward_value = -dG_over_sasa
    return reward_value, sequence, reward_dict