from typing import Tuple
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def charge_reward(
    pose,  # Individual pose passed by evaluator
    sequence: str,  # Individual sequence passed by evaluator
    design_chain: str,
    target_charge: float = 0.0,
    **kwargs
) -> Tuple[float, str, dict]:

    X = ProteinAnalysis(sequence)
    charge = X.charge_at_pH(7.0)
    reward = -abs(charge - target_charge)

    print(f"    charge={charge:.2f} (target={target_charge:.2f}) → reward={reward:.3f}")

    reward_dict = {
        'charge_reward': reward,
        'charge': charge,
        'target_charge': target_charge,
    }
    
    return reward, sequence, reward_dict
