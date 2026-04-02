# Version: v3.1  |  Date: 2026-04-02
"""
Parameter Scan Module
Sweeps a single parameter across a range of values, running batch simulations
at each standpoint to find where Moran's I stabilizes (convergence detection).
"""

import numpy as np
from dataclasses import dataclass, replace, field
from typing import List, Optional, Callable, Dict, Any
from simulation.engine import SimulationParams, SimulationResult, run_single_simulation
from simulation.statistics import calculate_all_statistics


SCANNABLE_PARAMS: Dict[str, Dict[str, Any]] = {
    'min_delta': {
        'label': 'Min Delta Angle',
        'min': 0.0,
        'max': 270.0,
        'default_step': 5.0,
        'default_start': 5.0,
        'default_stop': 20.0,
        'unit': '\u00b0',
        'description': 'Minimum random angle change on wall hit',
    },
    'max_delta': {
        'label': 'Max Delta Angle',
        'min': 0.0,
        'max': 270.0,
        'default_step': 5.0,
        'default_start': 5.0,
        'default_stop': 20.0,
        'unit': '\u00b0',
        'description': 'Maximum random angle change on wall hit',
    },
    'sample_interval': {
        'label': 'Sample Interval',
        'min': 1.0,
        'max': 60.0,
        'default_step': 5.0,
        'default_start': 5.0,
        'default_stop': 20.0,
        'unit': ' min',
        'description': 'Time between water samples',
    },
    'alpha': {
        'label': 'Initial Angle',
        'min': 0.0,
        'max': 360.0,
        'default_step': 5.0,
        'default_start': 5.0,
        'default_stop': 20.0,
        'unit': '\u00b0',
        'description': 'Initial heading direction',
    },
}


@dataclass
class ScanConfig:
    """Configuration for a parameter scan."""
    param_name: str
    start: float
    stop: float
    step: float
    runs_per_standpoint: int = 50
    base_seed: Optional[int] = 42


@dataclass
class ScanPointResult:
    """Results from batch runs at a single parameter standpoint."""
    param_value: float
    avg_morans_i_x: float
    avg_morans_i_y: float
    std_morans_i_x: float
    std_morans_i_y: float
    avg_coverage: float
    std_coverage: float
    num_runs: int


@dataclass
class ScanResult:
    """Complete result of a parameter scan."""
    config: ScanConfig
    fixed_params: SimulationParams
    points: List[ScanPointResult] = field(default_factory=list)

    # Computed after all points are collected
    param_values: Optional[np.ndarray] = None
    morans_i_x_values: Optional[np.ndarray] = None
    morans_i_y_values: Optional[np.ndarray] = None
    gradient_x: Optional[np.ndarray] = None
    gradient_y: Optional[np.ndarray] = None
    convergence_index_x: Optional[int] = None
    convergence_index_y: Optional[int] = None
    best_combined_index: Optional[int] = None

    def finalize(self):
        """Compute gradients, convergence, and best combined point after all points are collected."""
        if len(self.points) < 2:
            if len(self.points) == 1:
                self.best_combined_index = 0
            return

        self.param_values = np.array([p.param_value for p in self.points])
        self.morans_i_x_values = np.array([p.avg_morans_i_x for p in self.points])
        self.morans_i_y_values = np.array([p.avg_morans_i_y for p in self.points])

        self.gradient_x = compute_gradient(self.morans_i_x_values, self.param_values)
        self.gradient_y = compute_gradient(self.morans_i_y_values, self.param_values)

        self.convergence_index_x = detect_convergence(self.gradient_x)
        self.convergence_index_y = detect_convergence(self.gradient_y)

        combined = self.morans_i_x_values + self.morans_i_y_values
        self.best_combined_index = int(np.argmin(combined))


def compute_gradient(values: np.ndarray, param_values: np.ndarray) -> np.ndarray:
    """Compute the gradient (derivative) of values with respect to param_values."""
    return np.gradient(values, param_values)


def detect_convergence(gradient: np.ndarray, threshold_fraction: float = 0.1) -> Optional[int]:
    """
    Find the first index where the gradient stabilizes.

    The gradient is considered stable when |gradient[i]| drops below
    threshold_fraction * max(|gradient|) and stays below for all remaining points.

    Returns the index of the convergence point, or None if not found.
    """
    abs_grad = np.abs(gradient)
    peak = np.max(abs_grad)

    if peak < 1e-6:
        return 0

    threshold = threshold_fraction * peak

    for k in range(len(abs_grad)):
        if np.all(abs_grad[k:] < threshold):
            return k

    return None


def _validate_scan_params(base_params: SimulationParams, param_name: str, value: float) -> bool:
    """Check if a parameter value is valid given the other fixed params."""
    if param_name == 'min_delta' and value >= base_params.max_delta:
        return False
    if param_name == 'max_delta' and value <= base_params.min_delta:
        return False
    return True


def run_parameter_scan(
    base_params: SimulationParams,
    config: ScanConfig,
    standpoint_callback: Optional[Callable[[int, int, float, Optional[ScanPointResult]], None]] = None,
    run_callback: Optional[Callable[[int, int, int, int], None]] = None,
) -> ScanResult:
    """
    Sweep a single parameter across a range, running batch simulations at each standpoint.

    Args:
        base_params: Base simulation parameters (the scanned param will be overridden)
        config: Scan configuration (parameter, range, step, runs per standpoint)
        standpoint_callback: Called after each standpoint completes:
            standpoint_callback(current_standpoint, total_standpoints, param_value, point_result)
        run_callback: Called after each individual run within a standpoint:
            run_callback(standpoint_index, total_standpoints, current_run, total_runs)

    Returns:
        ScanResult with all points, gradients, and convergence indices
    """
    if config.param_name not in SCANNABLE_PARAMS:
        raise ValueError(f"Unknown scannable parameter: {config.param_name}")

    param_values = np.arange(config.start, config.stop + config.step * 0.5, config.step)
    total_standpoints = len(param_values)

    scan_result = ScanResult(config=config, fixed_params=base_params)

    for sp_idx, value in enumerate(param_values):
        if not _validate_scan_params(base_params, config.param_name, value):
            continue

        modified_params = replace(base_params, **{config.param_name: value})

        morans_x_list = []
        morans_y_list = []
        coverage_list = []

        for run_idx in range(config.runs_per_standpoint):
            seed = None if config.base_seed is None else config.base_seed + sp_idx * 1000 + run_idx

            result = run_single_simulation(modified_params, seed)
            result = calculate_all_statistics(result)

            morans_x_list.append(result.morans_i_x)
            morans_y_list.append(result.morans_i_y)
            coverage_list.append(result.coverage_percent)

            if run_callback:
                run_callback(sp_idx + 1, total_standpoints, run_idx + 1, config.runs_per_standpoint)

        point = ScanPointResult(
            param_value=value,
            avg_morans_i_x=float(np.mean(morans_x_list)),
            avg_morans_i_y=float(np.mean(morans_y_list)),
            std_morans_i_x=float(np.std(morans_x_list)),
            std_morans_i_y=float(np.std(morans_y_list)),
            avg_coverage=float(np.mean(coverage_list)),
            std_coverage=float(np.std(coverage_list)),
            num_runs=config.runs_per_standpoint,
        )
        scan_result.points.append(point)

        if standpoint_callback:
            standpoint_callback(sp_idx + 1, total_standpoints, value, point)

    scan_result.finalize()
    return scan_result
