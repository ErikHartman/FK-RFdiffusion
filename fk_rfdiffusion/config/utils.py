from pathlib import Path
from typing import  Optional
from omegaconf import OmegaConf, DictConfig
from ..utils import parse_chain_assignments
from ..feynman_kac.reward.configs import list_presets

def load_config_with_defaults(config_path: Optional[str] = None) -> DictConfig:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "base.yaml"
    
    conf = OmegaConf.load(config_path)
    return conf

def auto_detect_chain_assignments(conf: DictConfig, design_mode: str) -> DictConfig:
    if conf.reward.design_chain is None or (design_mode == 'binder' and conf.reward.target_chain is None):
        contigs = conf.contigmap.contigs
        if design_mode == 'binder':
            target_chain, design_chain = parse_chain_assignments(contigs)
            conf.reward.target_chain = target_chain
            conf.reward.design_chain = design_chain
        else:
            conf.reward.design_chain = 'A'
            conf.reward.target_chain = None
    
    return conf

def validate_config(conf: DictConfig) -> None:
    if conf.reward.function is not None:
        # Skip validation for custom reward functions (prefixed with "custom:")
        if not conf.reward.function.startswith("custom:"):
            available_presets = list_presets()
            if conf.reward.function not in available_presets:
                raise ValueError(f"Invalid reward preset: {conf.reward.function}. "
                               f"Available presets: {list(available_presets.keys())}")
    
    valid_modes = ['immediate', 'difference', 'max', 'sum', 'blind']
    if conf.feynman_kac.potential_mode not in valid_modes:
        raise ValueError(f"Invalid potential_mode: {conf.feynman_kac.potential_mode}. "
                        f"Valid options: {valid_modes}")