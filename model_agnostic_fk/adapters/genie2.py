"""Adapter for Genie 2's reverse-DDPM sampler.

The adapter delegates reward evaluation to the caller.  Existing FK rewards
remain in ``fk_rfdiffusion`` and are passed to the generic runner unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch

from ..interfaces import StepResult


@dataclass
class Genie2Particle:
    """One Genie 2 diffusion state and its conditioning features."""

    frames: Any
    features: dict[str, torch.Tensor]


class Genie2Adapter:
    """Expose Genie 2's reverse DDPM process to the generic FK loop.

    ``features_factory`` supports Genie 2's conditioned/motif sampling.  It
    returns Genie 2's unbatched NumPy feature dictionary, leaving all
    model-specific input construction outside the generic sampler.
    """

    def __init__(
        self,
        model: Any,
        *,
        length: int | None = None,
        sampling_noise_scale: float = 0.6,
        features_factory: Callable[[], dict[str, Any]] | None = None,
        final_timestep: int = 1,
    ) -> None:
        if (length is None) == (features_factory is None):
            raise ValueError("Specify exactly one of length or features_factory")
        if sampling_noise_scale < 0:
            raise ValueError("sampling_noise_scale must be non-negative")
        if final_timestep < 1:
            raise ValueError("final_timestep must be at least 1")

        self.model = model.eval()
        self.length = length
        self.features_factory = features_factory
        self.sampling_noise_scale = sampling_noise_scale
        self._final_timestep = final_timestep
        self.model.setup_schedule()

    @property
    def initial_timestep(self) -> int:
        return int(self.model.config.diffusion["n_timestep"])

    @property
    def final_timestep(self) -> int:
        return self._final_timestep

    def sample_initial_particle(self) -> Genie2Particle:
        from genie.utils.affine_utils import T
        from genie.utils.feat_utils import (
            batchify_np_features,
            convert_np_features_to_tensor,
            create_empty_np_features,
        )
        from genie.utils.geo_utils import compute_frenet_frames

        np_features = (
            create_empty_np_features([self.length])
            if self.features_factory is None
            else self.features_factory()
        )
        features = convert_np_features_to_tensor(
            batchify_np_features([np_features]), self.model.device
        )
        translations = torch.randn_like(features["atom_positions"])
        rotations = compute_frenet_frames(
            translations, features["chain_index"], features["residue_mask"]
        )
        return Genie2Particle(T(rotations, translations), features)

    def step(self, particle: Genie2Particle, *, timestep: int) -> StepResult[Genie2Particle]:
        from genie.utils.affine_utils import T
        from genie.utils.geo_utils import compute_frenet_frames

        if not self.final_timestep <= timestep <= self.initial_timestep:
            raise ValueError(f"Timestep {timestep} is outside the configured diffusion range")

        timesteps = torch.full(
            (particle.frames.trans.shape[0],), timestep, dtype=torch.int, device=self.model.device
        )
        with torch.no_grad():
            predicted_noise = self.model.model(particle.frames, timesteps, particle.features)["z"]

        # Genie predicts epsilon. Convert it to x0 for the FK reward, while
        # retaining its original posterior update for the next state.
        alpha_bar = self.model.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sigma_bar = self.model.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        denoised_translations = (particle.frames.trans - sigma_bar * predicted_noise) / alpha_bar
        denoised_translations *= particle.features["residue_mask"].unsqueeze(-1)
        denoised_rotations = compute_frenet_frames(
            denoised_translations, particle.features["chain_index"], particle.features["residue_mask"]
        )

        weight = (1.0 - self.model.alphas[timesteps]) / self.model.sqrt_one_minus_alphas_cumprod[timesteps]
        posterior_mean = (1.0 / self.model.sqrt_alphas[timesteps]).view(-1, 1, 1) * (
            particle.frames.trans - weight.view(-1, 1, 1) * predicted_noise
        )
        posterior_mean *= particle.features["residue_mask"].unsqueeze(-1)
        if timestep == self.final_timestep:
            next_translations = posterior_mean
        else:
            next_translations = posterior_mean + (
                self.sampling_noise_scale
                * self.model.sqrt_betas[timesteps].view(-1, 1, 1)
                * torch.randn_like(particle.frames.trans)
            )
            next_translations *= particle.features["residue_mask"].unsqueeze(-1)
        next_rotations = compute_frenet_frames(
            next_translations, particle.features["chain_index"], particle.features["residue_mask"]
        )
        return StepResult(
            denoised_particle=Genie2Particle(T(denoised_rotations, denoised_translations), particle.features),
            next_particle=Genie2Particle(T(next_rotations, next_translations), particle.features),
        )

    def clone_particle(self, particle: Genie2Particle) -> Genie2Particle:
        from genie.utils.affine_utils import T

        return Genie2Particle(
            T(particle.frames.rots.clone(), particle.frames.trans.clone()),
            {name: value.clone() for name, value in particle.features.items()},
        )

    def save_particle(
        self,
        particle: Genie2Particle,
        *,
        timestep: int,
        particle_id: str,
        output_directory: Path,
        kind: Literal["state", "denoised", "final"],
    ) -> Path:
        from genie.utils.feat_utils import (
            convert_tensor_features_to_numpy,
            debatchify_np_features,
            save_np_features_to_pdb,
        )

        destination = output_directory / "trajectories" / f"t_{timestep:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{'x0_' if kind == 'denoised' else ''}{particle_id}.pdb"
        features = {name: value.clone() for name, value in particle.features.items()}
        features["atom_positions"] = particle.frames.trans.detach().clone().cpu()
        np_features = debatchify_np_features(convert_tensor_features_to_numpy(features))[0]
        save_np_features_to_pdb(np_features, path)
        return path
