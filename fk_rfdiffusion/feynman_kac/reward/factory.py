from functools import partial
from omegaconf import DictConfig
from .configs import get_reward_preset
from .interface import dG_reward, dSASA_reward, dGdSASA_reward
from .secondary_structure import secondary_structure_reward
from .sequence import charge_reward
from .base import MultiSequenceEvaluator

def get_reward_function(conf: DictConfig):
    """
    Create a reward function from config. Returns a partial function that's picklable
    for multiprocessing by converting all config objects to primitive types.
    """
    function_name = str(conf.reward.function)
    reward_config = get_reward_preset(function_name)
    
    if hasattr(conf.reward, 'target_chain') and conf.reward.target_chain:
        reward_config['target_chain'] = str(conf.reward.target_chain)
    
    # Convert all config objects to plain primitive types for pickling
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
    
    # Handle multi-sequence evaluation parameters
    aggregation_mode = conf.reward.aggregation_mode
    n_sequences = conf.reward.n_sequences
    
    function_name = reward_config['function']
    
    # Create kwargs for the single-sequence function (excluding design_chain)
    single_seq_kwargs = {}
    
    # Select the appropriate single-sequence function and add specific parameters
    if function_name == 'interface_dG':
        single_seq_kwargs['target_chain'] = str(conf.reward.target_chain)
        single_seq_function = dG_reward
    elif function_name == 'interface_dGdSASA':
        single_seq_kwargs['target_chain'] = str(conf.reward.target_chain)
        single_seq_function = dGdSASA_reward
    elif function_name == 'interface_dSASA':
        single_seq_kwargs['target_chain'] = str(conf.reward.target_chain)
        single_seq_function = dSASA_reward
    elif function_name == 'secondary_structure':
        single_seq_kwargs['target_alpha'] = float(reward_config['target_alpha'])
        single_seq_kwargs['target_beta'] = float(reward_config['target_beta'])
        single_seq_kwargs['target_loop'] = float(reward_config['target_loop'])
        single_seq_kwargs['weight_alpha'] = float(reward_config['weight_alpha'])
        single_seq_kwargs['weight_beta'] = float(reward_config['weight_beta'])
        single_seq_kwargs['weight_loop'] = float(reward_config['weight_loop'])
        single_seq_function = secondary_structure_reward
    elif function_name == 'positive_charge' or function_name == 'negative_charge':
        single_seq_kwargs['target_charge'] = float(reward_config['target_charge'])
        single_seq_function = charge_reward
    else:
        raise ValueError(f"Unknown reward function type: {function_name}")
    
    # Create the evaluator (handles both single and multi-sequence cases)
    evaluator = MultiSequenceEvaluator(
        single_sequence_evaluator=partial(single_seq_function, **single_seq_kwargs),
        design_chain=str(conf.reward.design_chain),
        mpnn_config=mpnn_config,
        n_sequences=n_sequences,
        aggregation_mode=aggregation_mode,
        is_symmetric=bool(conf.inference.symmetry is not None and conf.inference.symmetry != '')
    )

    if conf.feynman_kac.tau is None:
        conf.feynman_kac.tau = float(reward_config["tau"])
        print("Using recommended tau from reward function: ", conf.feynman_kac.tau)
    
    return evaluator