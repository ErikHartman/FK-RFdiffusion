"""Run fixed alpha-helix and beta-sheet FK designs with Genie 2.

The two objectives reuse the existing FK reward factory.  This script has no
command-line parameters: adjust the constants below only when changing the
showcase configuration.
"""

from __future__ import annotations

import os

# PyTorch reads this at import time. Genie imports Torch below, so it must be
# configured before importing Genie (setting it inside main is too late).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fk_rfdiffusion.config.utils import load_config_with_defaults
from fk_rfdiffusion.feynman_kac.reward import ensure_pyrosetta_initialized, get_reward_function
from pathlib import Path

from genie.utils.model_io import load_pretrained_model
from model_agnostic_fk import run_feynman_kac
from model_agnostic_fk.adapters import Genie2Adapter


LENGTH = 120
N_PARTICLES = 5
N_TIMESTEPS = 1_000
GUIDANCE_START_TIMESTEP = 800
RESAMPLING_FREQUENCY = 100
SAMPLING_NOISE_SCALE = 0.6

EXAMPLES_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = EXAMPLES_DIRECTORY / "output"
CHECKPOINT_DIRECTORY = EXAMPLES_DIRECTORY.parent / "externals" / "Genie2" / "results"


def make_reward(reward_name: str):

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
    device = "mps"
    print(f"Running Genie 2 FK showcase on {device}.")
    model = load_pretrained_model(str(CHECKPOINT_DIRECTORY), "base", 40).to(device).eval()
    model.config.diffusion["n_timestep"] = N_TIMESTEPS
    print("Running helix")
    run_objective(model, "alpha_helix_ss", "alpha_helix")

    print("Running beta")
    run_objective(model, "beta_sheet_ss", "beta_sheet")


if __name__ == "__main__":
    main()
