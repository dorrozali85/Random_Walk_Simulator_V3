# Version: v3.3  |  Date: 2026-04-03
"""
CSV Export Module
Generates CSV files for single runs and batch runs per specification.
"""

import csv
import io
import os
from datetime import datetime
from typing import List
from simulation.engine import SimulationResult
from simulation.batch import BatchResult


def _format_params_section(params, scan_config=None) -> list:
    """
    Build rows describing all simulation parameters for appending to any CSV summary.
    Optionally includes sweep config if scan_config is provided.
    Returns a list of rows (each row is a list).
    """
    rows = []
    rows.append([])
    rows.append(['--- RUN PARAMETERS ---'])
    rows.append(['Pool Width (m)', params.pool_width])
    rows.append(['Pool Height (m)', params.pool_height])
    rows.append(['Initial Angle (deg)', params.alpha])
    rows.append(['Min Delta (deg)', params.min_delta])
    rows.append(['Max Delta (deg)', params.max_delta])
    rows.append(['Sample Interval (min)', params.sample_interval])
    rows.append(['Max Samples', params.max_samples])
    rows.append(['Cruise Speed (m/s)', params.cruise_speed])
    rows.append(['Slowdown Factor', params.slowdown_factor])
    rows.append(['Edge Buffer (m)', params.edge_buffer])
    rows.append(['Boat Width (m)', params.boat_width])
    rows.append(['Stop Time (s)', params.stop_time])
    rows.append(['Acceleration (m/s²)', params.acceleration])

    if scan_config is not None:
        rows.append([])
        rows.append(['--- SWEEP PARAMETERS ---'])
        rows.append(['Scanned Parameter', scan_config.param_name])
        rows.append(['Scan Start', scan_config.start])
        rows.append(['Scan Stop', scan_config.stop])
        rows.append(['Scan Step', scan_config.step])
        rows.append(['Runs per Standpoint', scan_config.runs_per_standpoint])

    return rows


def save_log_file(content: str, results_dir: str, run_type: str, num_iterations: int) -> str:
    """
    Save CSV content to results/ directory with a standardized filename.

    Filename format: YYYYMMDD-NNN-IIII-type.csv
      YYYYMMDD       - date of run
      NNN            - run number for today (3 digits, counts existing files)
      IIII           - total simulations executed (4 digits)
      type           - 'single', 'batch', or 'sweep'

    Returns the full path of the saved file.
    """
    os.makedirs(results_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    existing = [f for f in os.listdir(results_dir) if f.startswith(today) and f.endswith('.csv')]
    run_num = len(existing) + 1
    filename = f"{today}-{run_num:03d}-{num_iterations:04d}-{run_type}.csv"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        f.write(content)
    return filepath


def generate_single_run_csv(result: SimulationResult) -> str:
    """
    Generate CSV content for a single simulation run.

    Format:
    - Header: Timestamp,Event,PositionX,PositionY,AngleChange
    - Events: Start, WallHit, WaterSample
    - Summary section at end (including run parameters)
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Timestamp', 'Event', 'PositionX', 'PositionY', 'AngleChange'])

    # Write events
    for event in result.events:
        writer.writerow([
            f"{event.timestamp:.2f}",
            event.event_type,
            f"{event.position_x:.4f}",
            f"{event.position_y:.4f}",
            f"{event.angle_change:.2f}"
        ])

    # Write summary section
    writer.writerow([])
    writer.writerow(['--- SUMMARY ---'])
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Moran\'s I (X)', f"{result.morans_i_x:.4f}"])
    writer.writerow(['Moran\'s I (Y)', f"{result.morans_i_y:.4f}"])
    writer.writerow(['Coverage %', f"{result.coverage_percent:.2f}"])
    writer.writerow(['Min Distance (m)', f"{result.min_distance:.4f}"])
    writer.writerow(['Max Distance (m)', f"{result.max_distance:.4f}"])
    writer.writerow(['Avg Distance (m)', f"{result.avg_distance:.4f}"])
    writer.writerow(['Lag-1 Correlation', f"{result.lag1_correlation:.4f}"])
    writer.writerow(['Total Time (s)', f"{result.total_time:.2f}"])
    writer.writerow(['Wall Hits', result.num_wall_hits])
    writer.writerow(['Sample Count', len(result.samples)])

    for row in _format_params_section(result.params):
        writer.writerow(row)

    return output.getvalue()


def generate_batch_csv(batch_result: BatchResult) -> str:
    """
    Generate CSV content for batch runs.

    Format (long format):
    - Header: Run#,Timestamp,Event,PositionX,PositionY,AngleChange
    - All runs stacked vertically
    - Summary after each run
    - Blank line between runs
    - Batch aggregate summary + run parameters at end
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Run#', 'Timestamp', 'Event', 'PositionX', 'PositionY', 'AngleChange'])

    # Write each run
    for run_idx, result in enumerate(batch_result.runs, start=1):
        for event in result.events:
            writer.writerow([
                run_idx,
                f"{event.timestamp:.2f}",
                event.event_type,
                f"{event.position_x:.4f}",
                f"{event.position_y:.4f}",
                f"{event.angle_change:.2f}"
            ])

        writer.writerow([])
        writer.writerow([run_idx, '--- SUMMARY ---', '', '', '', ''])
        writer.writerow([run_idx, 'Moran\'s I (X)', f"{result.morans_i_x:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Moran\'s I (Y)', f"{result.morans_i_y:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Coverage %', f"{result.coverage_percent:.2f}", '', '', ''])
        writer.writerow([run_idx, 'Min Distance (m)', f"{result.min_distance:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Max Distance (m)', f"{result.max_distance:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Avg Distance (m)', f"{result.avg_distance:.4f}", '', '', ''])
        writer.writerow([])

    # Write batch aggregate statistics
    stats = batch_result.statistics
    writer.writerow(['=== BATCH SUMMARY ===', '', '', '', '', ''])
    writer.writerow(['Total Runs', stats.num_runs, '', '', '', ''])
    writer.writerow(['Avg Moran\'s I (X)', f"{stats.avg_morans_i_x:.4f}",
                     '± std', f"{stats.std_morans_i_x:.4f}", '', ''])
    writer.writerow(['Avg Moran\'s I (Y)', f"{stats.avg_morans_i_y:.4f}",
                     '± std', f"{stats.std_morans_i_y:.4f}", '', ''])
    writer.writerow(['Avg Coverage %', f"{stats.avg_coverage:.2f}",
                     '± std', f"{stats.std_coverage:.2f}", '', ''])
    writer.writerow(['Avg Min Distance (m)', f"{stats.avg_min_distance:.4f}", '', '', '', ''])
    writer.writerow(['Avg Max Distance (m)', f"{stats.avg_max_distance:.4f}", '', '', '', ''])
    writer.writerow(['Avg Avg Distance (m)', f"{stats.avg_avg_distance:.4f}", '', '', '', ''])

    for row in _format_params_section(batch_result.params):
        writer.writerow(row)

    return output.getvalue()


def generate_scan_csv(scan_result) -> str:
    """
    Generate CSV content for a parameter scan result.

    Format: one row per standpoint with Moran's I values, std, gradient, coverage.
    Includes convergence, best parameter, run parameters, and sweep config at end.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    config = scan_result.config
    param_info = f"{config.param_name} [{config.start} to {config.stop} step {config.step}]"

    writer.writerow([f'Parameter Scan: {param_info}'])
    writer.writerow([f'Runs per standpoint: {config.runs_per_standpoint}'])
    writer.writerow([])

    writer.writerow([
        'ParamValue', 'Avg_Morans_I_X', 'Std_Morans_I_X',
        'Avg_Morans_I_Y', 'Std_Morans_I_Y',
        'Combined_Score',
        'Gradient_X', 'Gradient_Y',
        'Avg_Coverage', 'Std_Coverage', 'Num_Runs'
    ])

    for i, point in enumerate(scan_result.points):
        grad_x = scan_result.gradient_x[i] if scan_result.gradient_x is not None else ''
        grad_y = scan_result.gradient_y[i] if scan_result.gradient_y is not None else ''
        combined = point.avg_morans_i_x + point.avg_morans_i_y
        writer.writerow([
            f"{point.param_value:.2f}",
            f"{point.avg_morans_i_x:.6f}",
            f"{point.std_morans_i_x:.6f}",
            f"{point.avg_morans_i_y:.6f}",
            f"{point.std_morans_i_y:.6f}",
            f"{combined:.6f}",
            f"{grad_x:.6f}" if isinstance(grad_x, float) else '',
            f"{grad_y:.6f}" if isinstance(grad_y, float) else '',
            f"{point.avg_coverage:.2f}",
            f"{point.std_coverage:.2f}",
            point.num_runs,
        ])

    writer.writerow([])
    writer.writerow(['--- CONVERGENCE ---'])
    if scan_result.convergence_index_x is not None:
        cv = scan_result.points[scan_result.convergence_index_x].param_value
        writer.writerow(['Convergence X', f"{cv:.2f}"])
    else:
        writer.writerow(['Convergence X', 'Not found'])
    if scan_result.convergence_index_y is not None:
        cv = scan_result.points[scan_result.convergence_index_y].param_value
        writer.writerow(['Convergence Y', f"{cv:.2f}"])
    else:
        writer.writerow(['Convergence Y', 'Not found'])

    writer.writerow([])
    writer.writerow(['--- BEST PARAMETER (Lowest Combined Moran\'s I) ---'])
    if scan_result.best_combined_index is not None:
        best = scan_result.points[scan_result.best_combined_index]
        combined = best.avg_morans_i_x + best.avg_morans_i_y
        writer.writerow(['Best Parameter Value', f"{best.param_value:.2f}"])
        writer.writerow(['Combined Score (X+Y)', f"{combined:.6f}"])
        writer.writerow(['Moran\'s I (X)', f"{best.avg_morans_i_x:.6f}"])
        writer.writerow(['Moran\'s I (Y)', f"{best.avg_morans_i_y:.6f}"])
        writer.writerow(['Coverage %', f"{best.avg_coverage:.2f}"])
    else:
        writer.writerow(['Best Parameter', 'Not found'])

    for row in _format_params_section(scan_result.fixed_params, scan_config=scan_result.config):
        writer.writerow(row)

    return output.getvalue()


def generate_convergence_csv(points, params, max_n: int, seed: int) -> str:
    """
    Generate CSV content for a convergence analysis run.

    Format:
    - Analysis config header (max_n, seed, null_baseline)
    - Data table: one row per checkpoint (N, mean, CI, SE for X and Y)
    - Verdict at final checkpoint (CI below null = proven)
    - Full simulation parameters section at end
    """
    output = io.StringIO()
    writer = csv.writer(output)

    null_b = points[0].null_baseline if points else 0.0

    writer.writerow(['=== CONVERGENCE ANALYSIS ==='])
    writer.writerow(['Max N (total runs)', max_n])
    writer.writerow(['Random Seed', seed])
    writer.writerow(['Null Baseline E[I]', f"{null_b:.6f}"])
    writer.writerow(['Checkpoints computed', len(points)])
    writer.writerow([])

    # Data table
    writer.writerow([
        'N',
        'Mean_I_X', 'CI_Lower_X', 'CI_Upper_X', 'SE_X', 'Std_X',
        'Mean_I_Y', 'CI_Lower_Y', 'CI_Upper_Y', 'SE_Y', 'Std_Y',
        'Null_Baseline', 'CI_X_Below_Null', 'CI_Y_Below_Null',
    ])
    for pt in points:
        ci_x_ok = 'YES' if pt.ci_upper_x < null_b else 'NO'
        ci_y_ok = 'YES' if pt.ci_upper_y < null_b else 'NO'
        writer.writerow([
            pt.n,
            f"{pt.mean_x:.6f}", f"{pt.ci_lower_x:.6f}", f"{pt.ci_upper_x:.6f}",
            f"{pt.se_x:.6f}",   f"{pt.std_x:.6f}",
            f"{pt.mean_y:.6f}", f"{pt.ci_lower_y:.6f}", f"{pt.ci_upper_y:.6f}",
            f"{pt.se_y:.6f}",   f"{pt.std_y:.6f}",
            f"{pt.null_baseline:.6f}",
            ci_x_ok, ci_y_ok,
        ])

    # Verdict at final checkpoint
    if points:
        last = points[-1]
        writer.writerow([])
        writer.writerow(['--- FINAL VERDICT (N =', last.n, ') ---'])
        writer.writerow(['Null Baseline', f"{null_b:.6f}"])
        writer.writerow(['Final Mean I(X)', f"{last.mean_x:.6f}",
                         '95% CI', f"[{last.ci_lower_x:.6f}, {last.ci_upper_x:.6f}]",
                         'Verdict', 'PROVEN (CI below null)' if last.ci_upper_x < null_b else 'NOT PROVEN'])
        writer.writerow(['Final Mean I(Y)', f"{last.mean_y:.6f}",
                         '95% CI', f"[{last.ci_lower_y:.6f}, {last.ci_upper_y:.6f}]",
                         'Verdict', 'PROVEN (CI below null)' if last.ci_upper_y < null_b else 'NOT PROVEN'])
        writer.writerow(['Final SE(X)', f"{last.se_x:.6f}",
                         'Final SE(Y)', f"{last.se_y:.6f}"])

    for row in _format_params_section(params):
        writer.writerow(row)

    return output.getvalue()


def get_csv_filename(batch: bool, num_runs: int = 1) -> str:
    """Generate appropriate filename for CSV export."""
    if batch and num_runs > 1:
        return f"simulation_batch_{num_runs}_runs.csv"
    else:
        return "simulation_single_run.csv"
