"""Simulation package for boat random walk simulation."""
from simulation.engine import SimulationParams, SimulationResult, BoatSimulator, run_single_simulation
from simulation.statistics import calculate_all_statistics, find_nearest_neighbors
from simulation.batch import BatchResult, BatchStatistics, run_batch_simulation
from simulation.parameter_scan import (
    ScanConfig, ScanPointResult, ScanResult, SCANNABLE_PARAMS, run_parameter_scan
)
from simulation.convergence_analysis import ConvergencePoint, run_convergence_analysis

__all__ = [
    'SimulationParams', 'SimulationResult', 'BoatSimulator', 'run_single_simulation',
    'calculate_all_statistics', 'find_nearest_neighbors',
    'BatchResult', 'BatchStatistics', 'run_batch_simulation',
    'ScanConfig', 'ScanPointResult', 'ScanResult', 'SCANNABLE_PARAMS', 'run_parameter_scan',
    'ConvergencePoint', 'run_convergence_analysis',
]
