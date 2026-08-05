"""Maximize the existing FK alpha-helix reward with Genie 2.

Run this from the repository root after following ``model_agnostic_fk/README``.
The reward implementation is imported directly from ``fk_rfdiffusion``: this
example contains no ProteinMPNN, PyRosetta, or reward logic of its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from genie.utils.model_io import load_pretrained_model
from model_agnostic_fk import run_feynman_kac
from model_agnostic_fk.adapters import Genie2Adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=50, help="Generated protein length")
    parser.add_argument("--particles", type=int, default=4, help="FK particle count")
    parser.add_argument("--timesteps", type=int, default=1000, help="Genie reverse-diffusion steps")
    parser.add_argument("--guidance-start", type=int, default=800)
    parser.add_argument("--resampling-frequency", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("designs/genie2_helix"))
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA, then MPS, then CPU")
    parser.add_argument(
        "--pyrosetta-site-packages",
        type=Path,
        help="Existing site-packages directory containing PyRosetta, such as a ColabFold environment",
    )
    return parser.parse_args()


def select_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    if args.timesteps < 2:
        raise ValueError("--timesteps must be at least 2")
    if not 1 < args.guidance_start <= args.timesteps:
        raise ValueError("--guidance-start must be between 2 and --timesteps")
    if args.pyrosetta_site_packages is not None:
        if not (args.pyrosetta_site_packages / "pyrosetta").is_dir():
            raise ValueError(f"No PyRosetta package found in {args.pyrosetta_site_packages}")
        # Keep this out of PYTHONPATH: ProteinMPNN is spawned below and must
        # continue to use the Genie environment's Torch/OpenMP libraries.
        sys.path.insert(0, str(args.pyrosetta_site_packages))

    # Reuse the current FK reward stack. Genie PDBs contain CA atoms, so this
    # chooses the CA-trained ProteinMPNN checkpoint; everything after sequence
    # design stays in the existing MultiSequenceEvaluator implementation.
    try:
        from fk_rfdiffusion.config.utils import load_config_with_defaults
        from fk_rfdiffusion.feynman_kac.reward import ensure_pyrosetta_initialized, get_reward_function
    except ModuleNotFoundError as error:
        if error.name == "pyrosetta":
            raise RuntimeError(
                "The shared FK reward requires PyRosetta. Activate the Genie environment, then run "
                "`python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'`."
            ) from error
        raise
    config = load_config_with_defaults()
    config.reward.function = "alpha_helix_ss"
    config.reward.design_chain = "A"
    config.reward.target_chain = None
    config.reward.n_sequences = 1
    config.mpnn.ca_only = True
    config.mpnn.use_soluble_model = False
    config.feynman_kac.tau = None
    ensure_pyrosetta_initialized()
    reward = get_reward_function(config)

    device = select_device(args.device)
    checkpoint_root = Path("model_agnostic_fk/externals/Genie2/results")
    model = load_pretrained_model(str(checkpoint_root), "base", 40).to(device).eval()
    # The schedule is parameterized by the requested inference trajectory; the
    # model itself retains its pretrained 1,000-step time embedding.
    model.config.diffusion["n_timestep"] = args.timesteps
    adapter = Genie2Adapter(model, length=args.length, sampling_noise_scale=0.6)

    results = run_feynman_kac(
        adapter,
        reward,
        output_directory=args.output,
        n_particles=args.particles,
        guidance_start_timestep=args.guidance_start,
        resampling_frequency=args.resampling_frequency,
        potential_mode="difference",
        tau=float(config.feynman_kac.tau),
        parallel_evaluation=False,
    )
    output_path = Path(args.output)
    results.to_csv(output_path / "results.csv", index=False)
    print(f"Saved {len(results)} FK evaluations to {output_path / 'results.csv'}")


if __name__ == "__main__":
    main()
