"""
Statistical Analysis Module
Computes Moran's I (spatial autocorrelation), coverage, and distance metrics.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import pdist, squareform
from simulation.engine import SimulationResult, SamplePoint, PathPoint


def calculate_morans_i(values: np.ndarray, positions: np.ndarray) -> float:
    """
    Calculate Moran's I spatial autocorrelation statistic.
    
    Moran's I measures spatial autocorrelation - how similar nearby observations are.
    Values range from -1 (perfect dispersion) to +1 (perfect clustering).
    Values near 0 indicate random spatial pattern.
    
    Args:
        values: 1D array of values to test for spatial autocorrelation
        positions: 2D array of (x, y) positions
        
    Returns:
        Moran's I statistic
    """
    n = len(values)
    if n < 2:
        return 0.0
    
    # Calculate spatial weights matrix (inverse distance)
    distances = squareform(pdist(positions))
    
    # Avoid division by zero - use small epsilon for self-distances
    np.fill_diagonal(distances, np.inf)
    
    # Inverse distance weights (row-standardized)
    weights = 1.0 / distances
    weights[~np.isfinite(weights)] = 0.0
    
    # Row-standardize weights
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums
    
    # Calculate Moran's I
    mean_val = np.mean(values)
    deviations = values - mean_val
    
    # Numerator: sum of weighted cross-products
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights[i, j] * deviations[i] * deviations[j]
    
    # Denominator: sum of squared deviations
    denominator = np.sum(deviations ** 2)
    
    if denominator == 0:
        return 0.0
    
    # Sum of weights
    W = np.sum(weights)
    
    if W == 0:
        return 0.0
    
    morans_i = (n / W) * (numerator / denominator)
    
    return morans_i


def calculate_coverage(path: List[PathPoint], pool_width: float, pool_height: float,
                       grid_cols: int = 20, grid_rows: int = 10) -> float:
    """
    Calculate coverage percentage - what fraction of the pool area was visited.
    
    Uses a grid overlay to determine which cells were traversed by the path.
    
    Args:
        path: List of path points
        pool_width: Pool width in meters
        pool_height: Pool height in meters
        grid_cols: Number of grid columns (default 20)
        grid_rows: Number of grid rows (default 10)
        
    Returns:
        Coverage percentage (0-100)
    """
    if not path:
        return 0.0
    
    # Create coverage grid
    cell_width = pool_width / grid_cols
    cell_height = pool_height / grid_rows
    visited = np.zeros((grid_rows, grid_cols), dtype=bool)
    
    # Mark cells visited by path
    for point in path:
        col = int(min(point.x / cell_width, grid_cols - 1))
        row = int(min(point.y / cell_height, grid_rows - 1))
        col = max(0, col)
        row = max(0, row)
        visited[row, col] = True
    
    # Also mark cells along line segments between consecutive points
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        _mark_line_cells(visited, p1.x, p1.y, p2.x, p2.y, 
                         cell_width, cell_height, grid_cols, grid_rows)
    
    # Calculate percentage
    total_cells = grid_rows * grid_cols
    visited_cells = np.sum(visited)
    
    return (visited_cells / total_cells) * 100.0


def _mark_line_cells(grid: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                     cell_width: float, cell_height: float, 
                     grid_cols: int, grid_rows: int) -> None:
    """Mark all grid cells that a line segment passes through."""
    # Use Bresenham-like approach with finer sampling
    steps = max(abs(x2 - x1) / cell_width, abs(y2 - y1) / cell_height)
    steps = max(int(steps * 2), 1)
    
    for t in np.linspace(0, 1, steps + 1):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        col = int(min(x / cell_width, grid_cols - 1))
        row = int(min(y / cell_height, grid_rows - 1))
        col = max(0, col)
        row = max(0, row)
        grid[row, col] = True


def calculate_sample_distances(samples: List[SamplePoint]) -> Tuple[float, float, float]:
    """
    Calculate min, max, and average distances between consecutive samples.
    
    Returns:
        Tuple of (min_distance, max_distance, avg_distance)
    """
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    
    distances = []
    for i in range(len(samples) - 1):
        s1, s2 = samples[i], samples[i + 1]
        dist = np.sqrt((s2.x - s1.x)**2 + (s2.y - s1.y)**2)
        distances.append(dist)
    
    return min(distances), max(distances), np.mean(distances)


def calculate_lag1_correlation(samples: List[SamplePoint]) -> float:
    """
    Calculate lag-1 autocorrelation of sample positions.
    
    Measures how correlated consecutive sample positions are.
    """
    if len(samples) < 3:
        return 0.0
    
    # Use distances from centroid
    xs = np.array([s.x for s in samples])
    ys = np.array([s.y for s in samples])
    
    # Combine x and y into distance from mean
    cx, cy = np.mean(xs), np.mean(ys)
    distances = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    
    if len(distances) < 2:
        return 0.0
    
    # Lag-1 correlation
    n = len(distances)
    mean_d = np.mean(distances)
    std_d = np.std(distances)
    
    if std_d == 0:
        return 0.0
    
    lag1_cov = np.sum((distances[:-1] - mean_d) * (distances[1:] - mean_d)) / (n - 1)
    lag1_corr = lag1_cov / (std_d ** 2)
    
    return lag1_corr


def calculate_all_statistics(result: SimulationResult) -> SimulationResult:
    """
    Calculate all statistics for a simulation result and update the result object.
    
    Args:
        result: SimulationResult with path and samples populated
        
    Returns:
        Updated SimulationResult with statistics filled in
    """
    samples = result.samples
    
    if len(samples) >= 2:
        # Extract positions
        positions = np.array([[s.x, s.y] for s in samples])
        xs = positions[:, 0]
        ys = positions[:, 1]
        
        # Moran's I for X and Y coordinates separately
        result.morans_i_x = calculate_morans_i(xs, positions)
        result.morans_i_y = calculate_morans_i(ys, positions)
        
        # Distance metrics
        result.min_distance, result.max_distance, result.avg_distance = \
            calculate_sample_distances(samples)
        
        # Lag-1 correlation
        result.lag1_correlation = calculate_lag1_correlation(samples)
    
    # Coverage (uses full path)
    result.coverage_percent = calculate_coverage(
        result.path,
        result.params.pool_width,
        result.params.pool_height
    )
    
    return result


def find_nearest_neighbors(samples: List[SamplePoint]) -> List[Tuple[int, int, float]]:
    """
    Find nearest neighbor for each sample point.
    
    Returns:
        List of tuples (from_idx, to_idx, distance)
    """
    if len(samples) < 2:
        return []
    
    positions = np.array([[s.x, s.y] for s in samples])
    distances = squareform(pdist(positions))
    
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(distances, np.inf)
    
    neighbors = []
    for i in range(len(samples)):
        nearest_idx = np.argmin(distances[i])
        nearest_dist = distances[i, nearest_idx]
        neighbors.append((i, nearest_idx, nearest_dist))
    
    return neighbors
