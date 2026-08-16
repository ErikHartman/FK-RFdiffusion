"""Small model-independent entry point for an FK design run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sampler import FeynmanKacSampler, RewardEvaluator


def run_feynman_kac(
    adapter: Any,
    reward_fn: RewardEvaluator,
    *,
    output_directory: str | Path,
    **sampler_options: Any,
):
    """Run FK with a model adapter and a supplied reward callable.

    The runner does not construct, wrap, or duplicate rewards. Pass the
    existing ``fk_rfdiffusion`` reward factory output directly.
    """

    return FeynmanKacSampler(
        adapter,
        reward_fn,
        output_directory=output_directory,
        **sampler_options,
    ).run()
