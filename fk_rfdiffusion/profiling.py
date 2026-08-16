"""Lightweight, opt-in wall-clock component profiling."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator


@dataclass
class TimingStat:
    calls: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    def record(self, elapsed_seconds: float) -> None:
        self.calls += 1
        self.total_seconds += elapsed_seconds
        self.min_seconds = min(self.min_seconds, elapsed_seconds)
        self.max_seconds = max(self.max_seconds, elapsed_seconds)

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0


class RuntimeProfiler:
    """Collect additive wall-clock timings for named operations.

    The profiler is process-local. It is intended for the serial timing
    benchmark and must not be used to aggregate ProcessPoolExecutor workers.
    """

    def __init__(self) -> None:
        self._stats: dict[str, TimingStat] = defaultdict(TimingStat)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self._stats[name].record(perf_counter() - started_at)

    def stat(self, name: str) -> TimingStat:
        return self._stats[name]

    def rows(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "component": name,
                "calls": stat.calls,
                "total_seconds": stat.total_seconds,
                "mean_seconds": stat.mean_seconds,
                "min_seconds": stat.min_seconds,
                "max_seconds": stat.max_seconds,
            }
            for name, stat in sorted(self._stats.items())
        ]
