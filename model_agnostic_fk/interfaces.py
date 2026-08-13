"""Contracts between the generic FK loop and a diffusion model.

An adapter owns every model-specific concern: tensor layout, conditioning
state, the denoising call, cloning, and conversion of a particle to an artifact
the reward function understands.  A particle may therefore be any Python
object, not necessarily coordinates or a PyTorch tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar


ParticleT = TypeVar("ParticleT")
ArtifactT = TypeVar("ArtifactT")


@dataclass(frozen=True)
class StepResult(Generic[ParticleT]):
    """The result of one reverse-diffusion step.

    ``denoised_particle`` is the model's x0 prediction, used for guidance.
    ``next_particle`` is the state to continue from at the next timestep.
    """

    denoised_particle: ParticleT
    next_particle: ParticleT


class DiffusionModelAdapter(Protocol[ParticleT, ArtifactT]):
    """Minimal interface required to run Feynman--Kac guidance.

    Each particle must contain all particle-local model state.  This matters
    for self-conditioning models: a child produced by resampling must not
    accidentally share a mutable cache with its parent or another particle.
    """

    @property
    def initial_timestep(self) -> int:
        """First reverse-diffusion timestep (for example, ``T``)."""

    @property
    def final_timestep(self) -> int:
        """Last reverse-diffusion timestep, normally ``1``."""

    def sample_initial_particle(self) -> ParticleT:
        """Create one independently sampled particle at ``initial_timestep``."""

    def step(self, particle: ParticleT, *, timestep: int) -> StepResult[ParticleT]:
        """Take one reverse-diffusion step for one particle."""

    def clone_particle(self, particle: ParticleT) -> ParticleT:
        """Return an independent copy suitable for a resampled child."""

    def save_particle(
        self,
        particle: ParticleT,
        *,
        timestep: int,
        particle_id: str,
        output_directory: Path,
        kind: Literal["state", "denoised", "final"],
    ) -> ArtifactT:
        """Persist a particle and return the artifact consumed by the reward."""
