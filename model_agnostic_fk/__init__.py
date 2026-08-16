"""Model-agnostic Feynman--Kac sampling for diffusion models.

The package deliberately has no dependency on a specific diffusion model.
Connect a model by implementing :class:`DiffusionModelAdapter`; Genie 2 is the
reference implementation in ``model_agnostic_fk.adapters``.
"""

from .interfaces import DiffusionModelAdapter, StepResult
from .runner import run_feynman_kac
from .sampler import FeynmanKacSampler

__all__ = ["DiffusionModelAdapter", "FeynmanKacSampler", "StepResult", "run_feynman_kac"]
