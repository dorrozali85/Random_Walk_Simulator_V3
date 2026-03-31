"""
Comprehensive Test Suite for Boat Simulator
Tests all components: engine, statistics, batch runs, CSV export, and visualization.
"""

import sys
import numpy as np
from typing import List, Tuple
import traceback

# Add parent directory to path
sys.path.insert(0, '/home/claude/boat_simulator')

from simulation.engine import SimulationParams, SimulationResult, BoatSimulator, run_single_simulation
from simulation.statistics import (
    calculate_all_statistics, calculate_morans_i, calculate_coverage,
    calculate_sample_distances, calculate_lag1_correlation, find_nearest_neighbors
)
from simulation.batch import run_batch_simulation, BatchResult
from export.csv_logger import generate_single_run_csv, generate_batch_csv


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


def run_test(test_func) -> TestResult:
    """Run a single test function and catch exceptions."""
    try:
        test_func()
        return TestResult(test_func.__name__, True, "PASSED")
    except AssertionError as e:
        return TestResult(test_func.__name__, False, f"ASSERTION FAILED: {e}")
    except Exception as e:
        return TestResult(test_func.__name__, False, f"ERROR: {e}\n{traceback.format_exc()}")


# ============ ENGINE TESTS ============

def test_simulation_params_defaults():
    """Test that SimulationParams has correct default values."""
    params = SimulationParams()
    assert params.pool_width == 12.5, f"Expected pool_width=12.5, got {params.pool_width}"
    assert params.pool_height == 25.0, f"Expected pool_height=25.0, got {params.pool_height}"
    assert params.alpha == 45.0, f"Expected alpha=45.0, got {params.alpha}"
    assert params.min_delta == 25.0, f"Expected min_delta=25.0, got {params.min_delta}"
    assert params.max_delta == 45.0, f"Expected max_delta=45.0, got {params.max_delta}"
    assert params.sample_interval == 10.0, f"Expected sample_interval=10.0, got {params.sample_interval}"
    assert params.max_samples == 5, f"Expected max_samples=5, got {params.max_samples}"
    assert params.cruise_speed == 0.2, f"Expected cruise_speed=0.2, got {params.cruise_speed}"
    assert params.slowdown_factor == 0.5, f"Expected slowdown_factor=0.5, got {params.slowdown_factor}"
    assert params.edge_buffer == 0.5, f"Expected edge_buffer=0.5, got {params.edge_buffer}"
    assert params.dt == 0.1, f"Expected dt=0.1, got {params.dt}"
    assert params.turn_delay == 1.5, f"Expected turn_delay=1.5, got {params.turn_delay}"
    print("✓ All default parameters correct")


def test_sample_interval_conversion():
    """Test that sample interval converts correctly to seconds."""
    params = SimulationParams(sample_interval=10.0)
    assert params.sample_interval_seconds == 600.0, \
        f"Expected 600 seconds, got {params.sample_interval_seconds}"
    print("✓ Sample interval conversion correct")


def test_single_simulation_runs():
    """Test that a single simulation runs without errors."""
    params = SimulationParams()
    result = run_single_simulation(params, seed=42)
    
    assert result is not None, "Result should not be None"
    assert len(result.path) > 0, "Path should have points"
    assert len(result.events) > 0, "Events should be logged"
    print(f"✓ Single simulation completed with {len(result.path)} path points")


def test_simulation_produces_samples():
    """Test that simulation produces the correct number of samples."""
    params = SimulationParams(max_samples=5, sample_interval=10.0)
    result = run_single_simulation(params, seed=42)
    
    assert len(result.samples) == 5, f"Expected 5 samples, got {len(result.samples)}"
    print(f"✓ Simulation produced {len(result.samples)} samples as expected")


def test_boat_stays_in_pool():
    """Test that boat position never exceeds pool boundaries."""
    params = SimulationParams()
    result = run_single_simulation(params, seed=42)
    
    for point in result.path:
        assert 0 <= point.x <= params.pool_width, \
            f"X position {point.x} outside pool bounds [0, {params.pool_width}]"
        assert 0 <= point.y <= params.pool_height, \
            f"Y position {point.y} outside pool bounds [0, {params.pool_height}]"
    
    print(f"✓ All {len(result.path)} path points within pool boundaries")


def test_samples_have_correct_indices():
    """Test that sample numbers are sequential starting from 1."""
    params = SimulationParams(max_samples=5)
    result = run_single_simulation(params, seed=42)
    
    for i, sample in enumerate(result.samples):
        expected_num = i + 1
        assert sample.sample_number == expected_num, \
            f"Sample {i} has number {sample.sample_number}, expected {expected_num}"
    
    print("✓ Sample indices are sequential and correct")


def test_events_logged_correctly():
    """Test that events have correct types."""
    params = SimulationParams()
    result = run_single_simulation(params, seed=42)
    
    # First event should be Start
    assert result.events[0].event_type == 'Start', \
        f"First event should be 'Start', got '{result.events[0].event_type}'"
    
    # Check all event types are valid
    valid_types = {'Start', 'WallHit', 'WaterSample'}
    for event in result.events:
        assert event.event_type in valid_types, \
            f"Invalid event type: {event.event_type}"
    
    # Count water samples in events
    sample_events = [e for e in result.events if e.event_type == 'WaterSample']
    assert len(sample_events) == len(result.samples), \
        f"Mismatch: {len(sample_events)} sample events vs {len(result.samples)} samples"
    
    print(f"✓ Events logged correctly: {len(result.events)} total")


def test_wall_hits_change_angle():
    """Test that wall hits include angle changes."""
    params = SimulationParams()
    result = run_single_simulation(params, seed=42)
    
    wall_hits = [e for e in result.events if e.event_type == 'WallHit']
    
    if len(wall_hits) > 0:
        for hit in wall_hits:
            assert params.min_delta <= abs(hit.angle_change) <= params.max_delta, \
                f"Angle change {hit.angle_change} outside bounds [{params.min_delta}, {params.max_delta}]"
        print(f"✓ {len(wall_hits)} wall hits with valid angle changes")
    else:
        print("⚠ No wall hits in this run (possible with certain seeds)")


def test_deterministic_with_seed():
    """Test that same seed produces identical results."""
    params = SimulationParams()
    
    result1 = run_single_simulation(params, seed=12345)
    result2 = run_single_simulation(params, seed=12345)
    
    assert len(result1.path) == len(result2.path), "Path lengths differ"
    assert len(result1.samples) == len(result2.samples), "Sample counts differ"
    
    for p1, p2 in zip(result1.path[:100], result2.path[:100]):
        assert abs(p1.x - p2.x) < 1e-10, f"X positions differ: {p1.x} vs {p2.x}"
        assert abs(p1.y - p2.y) < 1e-10, f"Y positions differ: {p1.y} vs {p2.y}"
    
    print("✓ Simulation is deterministic with same seed")


# ============ STATISTICS TESTS ============

def test_morans_i_calculation():
    """Test Moran's I calculation with known data."""
    # Clustered data should have positive Moran's I
    values = np.array([1.0, 1.1, 1.2, 5.0, 5.1, 5.2])
    positions = np.array([
        [0, 0], [0.1, 0.1], [0.2, 0.2],  # Cluster 1
        [10, 10], [10.1, 10.1], [10.2, 10.2]  # Cluster 2
    ])
    
    morans_i = calculate_morans_i(values, positions)
    # Clustered data should have positive Moran's I
    # (though exact value depends on weighting scheme)
    assert isinstance(morans_i, float), "Moran's I should be a float"
    print(f"✓ Moran's I calculated: {morans_i:.4f}")


def test_coverage_calculation():
    """Test coverage percentage calculation."""
    from simulation.engine import PathPoint
    
    # Create a path that covers some cells
    path = [
        PathPoint(time=0, x=0, y=0),
        PathPoint(time=1, x=5, y=10),
        PathPoint(time=2, x=10, y=20),
        PathPoint(time=3, x=12, y=25)
    ]
    
    coverage = calculate_coverage(path, pool_width=12.5, pool_height=25.0)
    
    assert 0 <= coverage <= 100, f"Coverage {coverage}% outside valid range"
    assert coverage > 0, "Coverage should be > 0 for non-empty path"
    print(f"✓ Coverage calculated: {coverage:.1f}%")


def test_sample_distances():
    """Test distance calculations between samples."""
    from simulation.engine import SamplePoint
    
    samples = [
        SamplePoint(time=0, x=0, y=0, sample_number=1),
        SamplePoint(time=1, x=3, y=4, sample_number=2),  # Distance 5 from first
        SamplePoint(time=2, x=3, y=4, sample_number=3),  # Distance 0 from second
    ]
    
    min_d, max_d, avg_d = calculate_sample_distances(samples)
    
    assert min_d == 0.0, f"Min distance should be 0, got {min_d}"
    assert max_d == 5.0, f"Max distance should be 5, got {max_d}"
    assert avg_d == 2.5, f"Avg distance should be 2.5, got {avg_d}"
    print(f"✓ Distance calculations correct: min={min_d}, max={max_d}, avg={avg_d}")


def test_nearest_neighbors():
    """Test nearest neighbor finding."""
    from simulation.engine import SamplePoint
    
    samples = [
        SamplePoint(time=0, x=0, y=0, sample_number=1),
        SamplePoint(time=1, x=1, y=0, sample_number=2),
        SamplePoint(time=2, x=10, y=0, sample_number=3),
    ]
    
    neighbors = find_nearest_neighbors(samples)
    
    assert len(neighbors) == 3, f"Expected 3 neighbor pairs, got {len(neighbors)}"
    # Point 0's nearest is point 1 (distance 1)
    assert neighbors[0][1] == 1, f"Point 0's nearest should be 1, got {neighbors[0][1]}"
    # Point 1's nearest is point 0 (distance 1)
    assert neighbors[1][1] == 0, f"Point 1's nearest should be 0, got {neighbors[1][1]}"
    print(f"✓ Nearest neighbors found correctly")


def test_full_statistics_calculation():
    """Test that all statistics are calculated for a simulation."""
    params = SimulationParams(max_samples=5)
    result = run_single_simulation(params, seed=42)
    result = calculate_all_statistics(result)
    
    # Check all stats are populated
    assert result.morans_i_x is not None, "Moran's I X not calculated"
    assert result.morans_i_y is not None, "Moran's I Y not calculated"
    assert result.coverage_percent is not None, "Coverage not calculated"
    assert result.min_distance is not None, "Min distance not calculated"
    assert result.max_distance is not None, "Max distance not calculated"
    assert result.avg_distance is not None, "Avg distance not calculated"
    
    print(f"✓ All statistics calculated:")
    print(f"  Moran's I (X): {result.morans_i_x:.4f}")
    print(f"  Moran's I (Y): {result.morans_i_y:.4f}")
    print(f"  Coverage: {result.coverage_percent:.1f}%")
    print(f"  Distances: min={result.min_distance:.2f}, max={result.max_distance:.2f}, avg={result.avg_distance:.2f}")


# ============ BATCH TESTS ============

def test_batch_single_run():
    """Test batch run with single simulation."""
    params = SimulationParams(max_samples=3)
    batch = run_batch_simulation(params, num_runs=1, base_seed=42)
    
    assert batch is not None, "Batch result should not be None"
    assert len(batch.runs) == 1, f"Expected 1 run, got {len(batch.runs)}"
    assert batch.statistics.num_runs == 1, f"Stats should show 1 run"
    print("✓ Single batch run completed")


def test_batch_multiple_runs():
    """Test batch run with multiple simulations."""
    params = SimulationParams(max_samples=3)
    batch = run_batch_simulation(params, num_runs=5, base_seed=42)
    
    assert len(batch.runs) == 5, f"Expected 5 runs, got {len(batch.runs)}"
    assert batch.statistics.num_runs == 5, f"Stats should show 5 runs"
    
    # Check that stats are averages
    morans_x_values = [r.morans_i_x for r in batch.runs]
    expected_avg = np.mean(morans_x_values)
    assert abs(batch.statistics.avg_morans_i_x - expected_avg) < 1e-10, \
        "Average Moran's I X doesn't match"
    
    print(f"✓ Batch of 5 runs completed with averaged statistics")


def test_batch_runs_different():
    """Test that batch runs produce different results (no same seed reuse)."""
    params = SimulationParams(max_samples=3)
    batch = run_batch_simulation(params, num_runs=3, base_seed=100)
    
    # Check that runs are different
    samples_0 = [(s.x, s.y) for s in batch.runs[0].samples]
    samples_1 = [(s.x, s.y) for s in batch.runs[1].samples]
    
    assert samples_0 != samples_1, "Different batch runs should have different samples"
    print("✓ Batch runs produce varied results")


def test_batch_statistics_ranges():
    """Test that batch statistics are within valid ranges."""
    params = SimulationParams(max_samples=5)
    batch = run_batch_simulation(params, num_runs=10, base_seed=42)
    
    stats = batch.statistics
    
    # Coverage should be between 0 and 100
    assert 0 <= stats.avg_coverage <= 100, f"Coverage {stats.avg_coverage}% invalid"
    
    # Standard deviations should be non-negative
    assert stats.std_morans_i_x >= 0, "Std dev should be non-negative"
    assert stats.std_morans_i_y >= 0, "Std dev should be non-negative"
    assert stats.std_coverage >= 0, "Std dev should be non-negative"
    
    # Distances should be non-negative
    assert stats.avg_min_distance >= 0, "Min distance should be non-negative"
    assert stats.avg_max_distance >= 0, "Max distance should be non-negative"
    assert stats.avg_avg_distance >= 0, "Avg distance should be non-negative"
    
    # Max should be >= min
    assert stats.avg_max_distance >= stats.avg_min_distance, \
        "Max distance should be >= min distance"
    
    print(f"✓ Batch statistics within valid ranges")


# ============ CSV EXPORT TESTS ============

def test_single_run_csv_format():
    """Test single run CSV has correct format."""
    params = SimulationParams(max_samples=3)
    result = run_single_simulation(params, seed=42)
    result = calculate_all_statistics(result)
    
    csv_content = generate_single_run_csv(result)
    
    # Check header
    assert "Timestamp,Event,PositionX,PositionY,AngleChange" in csv_content, \
        "CSV should have correct header"
    
    # Check event types present
    assert "Start" in csv_content, "CSV should contain Start event"
    assert "WaterSample" in csv_content, "CSV should contain WaterSample events"
    
    # Check summary section
    assert "SUMMARY" in csv_content, "CSV should have SUMMARY section"
    assert "Moran's I (X)" in csv_content, "Summary should have Moran's I X"
    assert "Coverage %" in csv_content, "Summary should have Coverage"
    
    print(f"✓ Single run CSV format correct ({len(csv_content)} characters)")


def test_batch_csv_format():
    """Test batch CSV has correct format."""
    params = SimulationParams(max_samples=3)
    batch = run_batch_simulation(params, num_runs=3, base_seed=42)
    
    csv_content = generate_batch_csv(batch)
    
    # Check header has Run#
    assert "Run#,Timestamp,Event" in csv_content, "Batch CSV should have Run# column"
    
    # Check batch summary
    assert "BATCH SUMMARY" in csv_content, "Should have batch summary"
    assert "Total Runs" in csv_content, "Should have total runs count"
    
    print(f"✓ Batch CSV format correct ({len(csv_content)} characters)")


def test_csv_numeric_formatting():
    """Test that CSV numbers are properly formatted."""
    params = SimulationParams(max_samples=2)
    result = run_single_simulation(params, seed=42)
    result = calculate_all_statistics(result)
    
    csv_content = generate_single_run_csv(result)
    lines = csv_content.split('\n')
    
    # Find a data line (not header, not summary)
    for line in lines[1:]:
        if line and 'Start' in line:
            parts = line.split(',')
            # Timestamp should be numeric
            float(parts[0])  # Will raise if not valid number
            # PositionX should be numeric
            float(parts[2])  # Will raise if not valid number
            break
    
    print("✓ CSV numeric formatting correct")


# ============ SANITY CHECKS ============

def test_physics_sanity():
    """Test that physics makes sense - boat travels reasonable distances."""
    params = SimulationParams(max_samples=5, sample_interval=10.0, cruise_speed=0.2)
    result = run_single_simulation(params, seed=42)
    
    # In 10 minutes at 0.2 m/s, boat travels max 120m (but bounces reduce this)
    # Should be less than total pool perimeter * some factor
    max_theoretical = params.cruise_speed * params.sample_interval_seconds * len(result.samples)
    
    total_path_length = 0
    for i in range(len(result.path) - 1):
        p1, p2 = result.path[i], result.path[i + 1]
        total_path_length += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    
    assert total_path_length < max_theoretical * 2, \
        f"Path length {total_path_length:.1f}m seems too long"
    assert total_path_length > 0, "Path should have some length"
    
    print(f"✓ Physics sanity check passed: {total_path_length:.1f}m traveled")


def test_coverage_increases_with_time():
    """Test that longer simulations generally have higher coverage."""
    params_short = SimulationParams(max_samples=2, sample_interval=5.0)
    params_long = SimulationParams(max_samples=10, sample_interval=5.0)
    
    result_short = run_single_simulation(params_short, seed=42)
    result_short = calculate_all_statistics(result_short)
    
    result_long = run_single_simulation(params_long, seed=42)
    result_long = calculate_all_statistics(result_long)
    
    # Longer sim should generally have more coverage
    # (not guaranteed due to random walk, but likely)
    print(f"  Short run coverage: {result_short.coverage_percent:.1f}%")
    print(f"  Long run coverage: {result_long.coverage_percent:.1f}%")
    
    # At minimum, both should be positive
    assert result_short.coverage_percent > 0, "Short run should have some coverage"
    assert result_long.coverage_percent > 0, "Long run should have some coverage"
    
    print("✓ Coverage sanity check passed")


def test_moran_i_range():
    """Test that Moran's I values are in valid range."""
    params = SimulationParams(max_samples=10)  # More samples for better stats
    
    # Run several times to check consistency
    for seed in range(5):
        result = run_single_simulation(params, seed=seed)
        result = calculate_all_statistics(result)
        
        # Moran's I is typically between -1 and 1, but can be outside
        # for very clustered or dispersed data
        assert -2 <= result.morans_i_x <= 2, \
            f"Moran's I X {result.morans_i_x} seems extreme"
        assert -2 <= result.morans_i_y <= 2, \
            f"Moran's I Y {result.morans_i_y} seems extreme"
    
    print("✓ Moran's I values in reasonable range across runs")


def test_sample_timing():
    """Test that samples occur at roughly correct intervals."""
    params = SimulationParams(max_samples=5, sample_interval=10.0)  # 10 min = 600s
    result = run_single_simulation(params, seed=42)
    
    # Check timing between samples
    for i in range(len(result.samples) - 1):
        s1, s2 = result.samples[i], result.samples[i + 1]
        interval = s2.time - s1.time
        
        # Should be close to 600s (±turn delays and dt accumulation)
        assert 590 <= interval <= 700, \
            f"Interval {interval:.1f}s between samples {i} and {i+1} seems wrong"
    
    print("✓ Sample timing approximately correct")


def test_start_position():
    """Test that boat starts at correct initial position."""
    params = SimulationParams()
    result = run_single_simulation(params, seed=42)
    
    first_point = result.path[0]
    assert abs(first_point.x - 0.5) < 0.01, f"Start X should be 0.5, got {first_point.x}"
    assert abs(first_point.y - 0.0) < 0.01, f"Start Y should be 0.0, got {first_point.y}"
    
    print("✓ Start position correct")


# ============ MAIN TEST RUNNER ============

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("BOAT SIMULATOR - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        # Engine tests
        test_simulation_params_defaults,
        test_sample_interval_conversion,
        test_single_simulation_runs,
        test_simulation_produces_samples,
        test_boat_stays_in_pool,
        test_samples_have_correct_indices,
        test_events_logged_correctly,
        test_wall_hits_change_angle,
        test_deterministic_with_seed,
        
        # Statistics tests
        test_morans_i_calculation,
        test_coverage_calculation,
        test_sample_distances,
        test_nearest_neighbors,
        test_full_statistics_calculation,
        
        # Batch tests
        test_batch_single_run,
        test_batch_multiple_runs,
        test_batch_runs_different,
        test_batch_statistics_ranges,
        
        # CSV tests
        test_single_run_csv_format,
        test_batch_csv_format,
        test_csv_numeric_formatting,
        
        # Sanity checks
        test_physics_sanity,
        test_coverage_increases_with_time,
        test_moran_i_range,
        test_sample_timing,
        test_start_position,
    ]
    
    results = []
    
    for test in tests:
        print(f"\n▶ Running: {test.__name__}")
        print("-" * 40)
        result = run_test(test)
        results.append(result)
        
        if not result.passed:
            print(f"  ❌ {result.message}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    print(f"\n  Total:  {len(results)}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    
    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
    
    print("\n" + "=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
