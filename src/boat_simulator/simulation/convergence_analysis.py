# Version: v3.3  |  Date: 2026-04-03
"""
Convergence Analysis Module
Runs increasing-N batches at a fixed parameter set to determine when
Moran's I estimates stabilise, proving statistical confidence.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
from simulation.engine import SimulationParams, run_single_simulation
from simulation.statistics import calculate_all_statistics


@dataclass
class ConvergencePoint:
    """Statistics computed on the first N runs of a convergence sequence."""
    n: int
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float
    se_x: float          # standard error = std / sqrt(n)
    se_y: float
    ci_lower_x: float    # 95% CI lower bound
    ci_upper_x: float
    ci_lower_y: float
    ci_upper_y: float
    null_baseline: float  # E[I] = -1 / (n_samples - 1) under complete spatial randomness


def _build_checkpoints(max_n: int) -> List[int]:
    """Return a geometric-ish sequence of N checkpoints up to max_n."""
    base = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200,
            300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
    result = [n for n in base if n <= max_n]
    if not result or result[-1] < max_n:
        result.append(max_n)
    return result


def run_convergence_analysis(
    params: SimulationParams,
    max_n: int = 200,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[ConvergencePoint]:
    """
    Run `max_n` simulations and compute Moran's I statistics at increasing
    N checkpoints.  Returns one ConvergencePoint per checkpoint.

    null_baseline = E[I] under complete spatial randomness = -1/(n_samples-1).
    When a checkpoint's ci_upper < null_baseline the result is statistically
    significantly better than random (95% confidence).
    """
    null_baseline = -1.0 / max(params.max_samples - 1, 1)

    # Run all simulations once, collecting raw Moran's I values
    raw_x: List[float] = []
    raw_y: List[float] = []

    for i in range(max_n):
        r = run_single_simulation(params, seed=seed + i)
        calculate_all_statistics(r)
        raw_x.append(r.morans_i_x)
        raw_y.append(r.morans_i_y)
        if progress_callback:
            progress_callback(i + 1, max_n)

    # Compute statistics at each checkpoint
    checkpoints = _build_checkpoints(max_n)
    points: List[ConvergencePoint] = []

    for n in checkpoints:
        sx = raw_x[:n]
        sy = raw_y[:n]

        mean_x = float(np.mean(sx))
        mean_y = float(np.mean(sy))
        std_x  = float(np.std(sx, ddof=1)) if n > 1 else 0.0
        std_y  = float(np.std(sy, ddof=1)) if n > 1 else 0.0
        se_x   = std_x / np.sqrt(n) if n > 1 else 0.0
        se_y   = std_y / np.sqrt(n) if n > 1 else 0.0

        points.append(ConvergencePoint(
            n=n,
            mean_x=mean_x, mean_y=mean_y,
            std_x=std_x,   std_y=std_y,
            se_x=se_x,     se_y=se_y,
            ci_lower_x=mean_x - 1.96 * se_x,
            ci_upper_x=mean_x + 1.96 * se_x,
            ci_lower_y=mean_y - 1.96 * se_y,
            ci_upper_y=mean_y + 1.96 * se_y,
            null_baseline=null_baseline,
        ))

    return points
