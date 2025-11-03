from typing import Dict, Any
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_reward_presets() -> DictConfig:
    """Load all reward presets from presets.yaml."""
    presets_path = Path(__file__).parent / "presets.yaml"
    return OmegaConf.load(presets_path)

def get_reward_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a specific reward preset configuration.
    """
    presets = load_reward_presets()
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown reward preset '{preset_name}'. Available presets: {available}")
    
    return OmegaConf.to_container(presets[preset_name], resolve=True)

def list_presets() -> Dict[str, str]:
    """
    List all available presets with their descriptions.
    """
    presets = load_reward_presets()
    preset_info = {}
    
    for name, config in presets.items():
        description = config.get('description', 'No description available')
        preset_info[name] = description
    
    return preset_info



