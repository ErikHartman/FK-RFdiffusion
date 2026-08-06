"""Run fixed positive- and negative-charge FK designs with Genie 2."""

from __future__ import annotations

import os
from pathlib import Path

# Genie imports Torch below. Configure MPS fallback before that import so
# unsupported MPS operations execute on CPU instead of aborting the run.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fk_rfdiffusion.config.utils import load_config_with_defaults
from fk_rfdiffusion.feynman_kac.reward import ensure_pyrosetta_initialized, get_reward_function
from genie.utils.model_io import load_pretrained_model
from model_agnostic_fk import run_feynman_kac
from model_agnostic_fk.adapters import Genie2Adapter


LENGTH = 30
N_PARTICLES = 10
N_TIMESTEPS = 1_000
GUIDANCE_START_TIMESTEP = 800
RESAMPLING_FREQUENCY = 50
SAMPLING_NOISE_SCALE = 0.6
DEVICE = "mps"

EXAMPLES_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = EXAMPLES_DIRECTORY / "output"
CHECKPOINT_DIRECTORY = EXAMPLES_DIRECTORY.parent / "externals" / "Genie2" / "results"


def make_reward(reward_name: str):
    """Create an unchanged FK charge reward with CA-only ProteinMPNN."""
    config = load_config_with_defaults()
    config.reward.function = reward_name
    config.reward.design_chain = "A"
    config.reward.target_chain = None
    config.reward.n_sequences = 1
    config.mpnn.ca_only = True
    config.mpnn.use_soluble_model = False
    config.feynman_kac.tau = None
    ensure_pyrosetta_initialized()
    return get_reward_function(config), float(config.feynman_kac.tau)


def run_objective(model, reward_name: str, output_name: str) -> None:
    reward, tau = make_reward(reward_name)
    adapter = Genie2Adapter(
        model,
        length=LENGTH,
        sampling_noise_scale=SAMPLING_NOISE_SCALE,
    )
    output_path = OUTPUT_DIRECTORY / output_name
    results = run_feynman_kac(
        adapter,
        reward,
        output_directory=output_path,
        n_particles=N_PARTICLES,
        guidance_start_timestep=GUIDANCE_START_TIMESTEP,
        resampling_frequency=RESAMPLING_FREQUENCY,
        potential_mode="immediate",
        tau=tau,
        parallel_evaluation=False,
    )
    results.to_csv(output_path / "results.csv", index=False)
    print(f"Saved {len(results)} FK evaluations to {output_path / 'results.csv'}")


def main() -> None:
    print(f"Running Genie 2 charge showcase on {DEVICE}.")
    model = load_pretrained_model(str(CHECKPOINT_DIRECTORY), "base", 40).to(DEVICE).eval()
    model.config.diffusion["n_timestep"] = N_TIMESTEPS

    print("Running positive-charge design (target charge +5).")
    run_objective(model, "positive_charge", "positive_charge")
    print("Running negative-charge design (target charge -10).")
    run_objective(model, "negative_charge", "negative_charge")


if __name__ == "__main__":
    main()
