"""A diffusion-model-independent Feynman--Kac particle filter."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar
import datetime

import numpy as np
import pandas as pd

from .interfaces import ArtifactT, DiffusionModelAdapter, ParticleT


RewardResult = tuple[float, str, dict[str, Any]]
RewardEvaluator = Callable[[ArtifactT], RewardResult]


class FeynmanKacSampler(Generic[ParticleT, ArtifactT]):
    """Guide any reverse-diffusion model by particle filtering.

    ``reward_fn`` receives the artifact returned by the adapter.  For protein
    models that will usually be a PDB path, but it can equally be a structure
    object, an mmCIF path, or model-native serialized output.
    """

    def __init__(
        self,
        adapter: DiffusionModelAdapter[ParticleT, ArtifactT],
        reward_fn: RewardEvaluator[ArtifactT],
        *,
        n_particles: int = 10,
        output_directory: str | Path = "./design",
        resampling_frequency: int = 5,
        guidance_start_timestep: int | None = None,
        save_full_trajectory: bool = False,
        parallel_evaluation: bool = False,
        max_workers: int | None = None,
        tau: float = 10.0,
        potential_mode: Literal["immediate", "difference", "max", "sum", "blind"] = "immediate",
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_particles < 1:
            raise ValueError("n_particles must be at least 1")
        if resampling_frequency < 1:
            raise ValueError("resampling_frequency must be at least 1")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if potential_mode not in {"immediate", "difference", "max", "sum", "blind"}:
            raise ValueError(f"Unknown potential_mode: {potential_mode}")

        self.adapter = adapter
        self.reward_fn = reward_fn
        self.n_particles = n_particles
        self.output_directory = Path(output_directory)
        self.resampling_frequency = resampling_frequency
        self.guidance_start_timestep = (
            adapter.initial_timestep if guidance_start_timestep is None else guidance_start_timestep
        )
        self.save_full_trajectory = save_full_trajectory
        self.parallel_evaluation = parallel_evaluation
        self.max_workers = max_workers
        self.tau = tau
        self.potential_mode = potential_mode
        self.rng = rng or np.random.default_rng()
        self._particle_counter = 0
        self._reward_history: dict[str, list[float]] = {}
        self._metadata: list[dict[str, Any]] = []

    def run(self) -> pd.DataFrame:
        """Run FK sampling and return one metadata row per evaluated particle."""
        if self.adapter.final_timestep > self.adapter.initial_timestep:
            raise ValueError("adapter.final_timestep must not exceed adapter.initial_timestep")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        start = datetime.datetime.now()
        particles = [self.adapter.sample_initial_particle() for _ in range(self.n_particles)]
        names = [self._new_particle_name() for _ in particles]
        parents: list[str | None] = [None] * self.n_particles

        for timestep in range(self.adapter.initial_timestep, self.adapter.final_timestep - 1, -1):
            if self.save_full_trajectory and timestep > self.adapter.final_timestep:
                for particle, name in zip(particles, names):
                    self._save(particle, timestep, name, "state")

            steps = [self.adapter.step(particle, timestep=timestep) for particle in particles]
            next_particles = [step.next_particle for step in steps]
            should_resample = (
                timestep <= self.guidance_start_timestep
                and timestep > self.adapter.final_timestep
                and timestep % self.resampling_frequency == 0
            )
            if should_resample:
                artifacts = [
                    self._save(step.denoised_particle, timestep, name, "denoised")
                    for step, name in zip(steps, names)
                ]
                rewards, sequences, details = self._evaluate(artifacts)
                self._record_rewards(names, rewards)
                selected, potentials = self._select(rewards, names)
                self._record_metadata(timestep, names, parents, rewards, potentials, sequences, details, artifacts)

                old_names = names
                particles = [self.adapter.clone_particle(next_particles[index]) for index in selected]
                parents = [old_names[index] for index in selected]
                names = [self._new_particle_name() for _ in selected]
                for child, parent in zip(names, parents):
                    self._reward_history[child] = self._reward_history[parent].copy()
            else:
                particles = next_particles

        artifacts = [
            self._save(particle, self.adapter.final_timestep, name, "final")
            for particle, name in zip(particles, names)
        ]
        rewards, sequences, details = self._evaluate(artifacts)
        self._record_rewards(names, rewards)
        self._record_metadata(
            self.adapter.final_timestep, names, parents, rewards, [None] * len(names), sequences, details, artifacts
        )
        elapsed = (datetime.datetime.now() - start).total_seconds()
        print(f"FK sampling completed in {elapsed:.1f} seconds")
        return pd.DataFrame(self._metadata)

    def _new_particle_name(self) -> str:
        name = f"p{self._particle_counter:04d}"
        self._particle_counter += 1
        return name

    def _save(self, particle: ParticleT, timestep: int, name: str, kind: Literal["state", "denoised", "final"]) -> ArtifactT:
        return self.adapter.save_particle(
            particle,
            timestep=timestep,
            particle_id=name,
            output_directory=self.output_directory,
            kind=kind,
        )

    def _evaluate(self, artifacts: Sequence[ArtifactT]) -> tuple[list[float], list[str], list[dict[str, Any]]]:
        if not self.parallel_evaluation:
            results = [self.reward_fn(artifact) for artifact in artifacts]
        else:
            # Process-based evaluation matches the existing PyRosetta workflow.
            # The reward callable and artifact type must be pickleable in this mode.
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.reward_fn, artifact): index for index, artifact in enumerate(artifacts)}
                ordered: list[RewardResult | None] = [None] * len(artifacts)
                for future in as_completed(futures):
                    ordered[futures[future]] = future.result()
            results = [result for result in ordered if result is not None]
        rewards, sequences, details = zip(*results)
        return [float(reward) for reward in rewards], list(sequences), list(details)

    def _record_rewards(self, names: Sequence[str], rewards: Sequence[float]) -> None:
        for name, reward in zip(names, rewards):
            self._reward_history.setdefault(name, []).append(reward)

    def _select(self, rewards: Sequence[float], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(rewards, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Reward function returned a non-finite value")

        if self.potential_mode == "blind":
            return np.arange(self.n_particles), np.ones(self.n_particles)
        if self.potential_mode == "immediate":
            score = values
        elif self.potential_mode == "difference":
            # The first actual guidance step can be earlier than the configured
            # start when the resampling frequency does not divide it.
            score = np.array([
                0.0 if len(self._reward_history[name]) < 2 else reward - self._reward_history[name][-2]
                for name, reward in zip(names, values)
            ])
        elif self.potential_mode == "max":
            score = np.array([max(self._reward_history[name]) for name in names])
        else:  # sum: use the historical mean, matching the current implementation's scale control
            score = np.array([np.mean(self._reward_history[name]) for name in names])

        log_weights = score / self.tau
        log_weights -= np.max(log_weights)
        potentials = np.exp(np.clip(log_weights, -1_000, 0))
        probabilities = potentials / potentials.sum()
        selected = self.rng.choice(len(values), size=self.n_particles, replace=True, p=probabilities)
        return selected, potentials

    def _record_metadata(
        self,
        timestep: int,
        names: Sequence[str],
        parents: Sequence[str | None],
        rewards: Sequence[float],
        potentials: Sequence[float | None],
        sequences: Sequence[str],
        details: Sequence[dict[str, Any]],
        artifacts: Sequence[ArtifactT],
    ) -> None:
        for name, parent, reward, potential, sequence, detail, artifact in zip(
            names, parents, rewards, potentials, sequences, details, artifacts
        ):
            self._metadata.append({
                "iteration": timestep,
                "particle_name": name,
                "parent_particle": parent,
                "reward": reward,
                "potential_value": potential,
                "potential_mode": self.potential_mode,
                "sequence": sequence,
                "artifact": str(artifact),
                **detail,
            })
