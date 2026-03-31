"""
Batch Runner Module
Orchestrates multiple simulation runs and aggregates statistics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
from simulation.engine import SimulationParams, SimulationResult, run_single_simulation
from simulation.statistics import calculate_all_statistics


@dataclass
class BatchStatistics:
    """Aggregated statistics from batch runs."""
    num_runs: int = 0
    avg_morans_i_x: float = 0.0
    avg_morans_i_y: float = 0.0
    avg_coverage: float = 0.0
    avg_min_distance: float = 0.0
    avg_max_distance: float = 0.0
    avg_avg_distance: float = 0.0
    
    # Standard deviations for error analysis
    std_morans_i_x: float = 0.0
    std_morans_i_y: float = 0.0
    std_coverage: float = 0.0


@dataclass
class BatchResult:
    """Complete result of a batch run."""
    params: SimulationParams
    runs: List[SimulationResult]
    statistics: BatchStatistics


def run_batch_simulation(
    params: SimulationParams,
    num_runs: int,
    base_seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> BatchResult:
    """
    Run multiple simulations with the same parameters.
    
    Args:
        params: Simulation parameters
        num_runs: Number of runs to execute
        base_seed: Optional base seed (each run uses base_seed + run_index)
        progress_callback: Optional callback(current_run, total_runs) for progress
        
    Returns:
        BatchResult with all runs and aggregated statistics
    """
    runs: List[SimulationResult] = []
    
    for i in range(num_runs):
        # Determine seed for this run
        seed = None if base_seed is None else base_seed + i
        
        # Run simulation
        result = run_single_simulation(params, seed)
        
        # Calculate statistics
        result = calculate_all_statistics(result)
        
        runs.append(result)
        
        # Report progress
        if progress_callback:
            progress_callback(i + 1, num_runs)
    
    # Aggregate statistics
    stats = _aggregate_statistics(runs)
    
    return BatchResult(params=params, runs=runs, statistics=stats)


def _aggregate_statistics(runs: List[SimulationResult]) -> BatchStatistics:
    """Calculate aggregate statistics from multiple runs."""
    if not runs:
        return BatchStatistics()
    
    n = len(runs)
    
    # Extract arrays
    morans_x = np.array([r.morans_i_x for r in runs])
    morans_y = np.array([r.morans_i_y for r in runs])
    coverage = np.array([r.coverage_percent for r in runs])
    min_dist = np.array([r.min_distance for r in runs])
    max_dist = np.array([r.max_distance for r in runs])
    avg_dist = np.array([r.avg_distance for r in runs])
    
    return BatchStatistics(
        num_runs=n,
        avg_morans_i_x=np.mean(morans_x),
        avg_morans_i_y=np.mean(morans_y),
        avg_coverage=np.mean(coverage),
        avg_min_distance=np.mean(min_dist),
        avg_max_distance=np.mean(max_dist),
        avg_avg_distance=np.mean(avg_dist),
        std_morans_i_x=np.std(morans_x),
        std_morans_i_y=np.std(morans_y),
        std_coverage=np.std(coverage)
    )
