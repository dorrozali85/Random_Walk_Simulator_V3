# Version: v3.3  |  Date: 2026-04-04
"""
Matrix Runner - execute a CSV-defined sequence of simulation jobs from the CLI.

Usage:
    python run_matrix.py <matrix_file.csv>

The CSV has one header row and one row per job. Empty cells inherit from a
DEFAULTS row (type=defaults) if present, otherwise use SimulationParams defaults.

Columns:
  label                - descriptive name for the job
  type                 - sweep | batch | convergence | defaults
  param_name           - (sweep only) which parameter to scan
  start, stop, step    - (sweep only) scan range
  runs_per_standpoint  - (sweep only) runs per standpoint
  num_runs             - (batch only) total batch runs
  max_n                - (convergence only) total sims
  seed                 - random seed (default 42)
  pool_width, pool_height, alpha, min_delta, max_delta,
  cruise_speed, slowdown_factor, edge_buffer, boat_width,
  stop_time, acceleration, sample_interval, max_samples
                       - any SimulationParams field (empty = use default)

Results (CSV + PNG) are saved to results/ using the same auto-save protocol
as the GUI.
"""

import sys
import os
import csv
import time

# Add the boat_simulator package to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'boat_simulator'))

from simulation.engine import SimulationParams, run_single_simulation
from simulation.statistics import calculate_all_statistics
from simulation.batch import run_batch_simulation
from simulation.parameter_scan import ScanConfig, run_parameter_scan
from simulation.convergence_analysis import run_convergence_analysis
from export.csv_logger import (
    generate_batch_csv, generate_scan_csv, generate_convergence_csv,
    save_log_file,
)
from visualization.figures import (
    save_screenshot,
    build_run_screenshot_figure,
    build_sweep_screenshot_figure,
    build_convergence_screenshot_figure,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# -- Helpers ----------------------------------------------------------------

_PARAM_FIELDS = {f.name for f in SimulationParams.__dataclass_fields__.values()}

# Fields that should be parsed as int (not float)
_INT_FIELDS = {'max_samples', 'runs_per_standpoint', 'num_runs', 'max_n', 'seed'}


def _parse_value(key, val_str):
    """Parse a CSV cell value to the correct Python type."""
    if key in _INT_FIELDS:
        return int(float(val_str))
    try:
        return float(val_str)
    except ValueError:
        return val_str


def row_to_dict(row: dict) -> dict:
    """Convert a CSV row dict to a clean dict, skipping empty cells."""
    result = {}
    for k, v in row.items():
        if k is None or v is None:
            continue
        if isinstance(v, list):
            continue  # skip extra columns from trailing commas
        k = k.strip()
        v = v.strip()
        if k and v:
            result[k] = _parse_value(k, v)
    return result


def params_from_dict(d: dict) -> SimulationParams:
    """Build a SimulationParams from a dict, ignoring unknown keys."""
    filtered = {k: v for k, v in d.items() if k in _PARAM_FIELDS}
    return SimulationParams(**filtered)


def merge_dicts(*dicts):
    """Merge multiple dicts left-to-right (later overrides earlier)."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


# -- Job runners ------------------------------------------------------------

def run_sweep_job(job, params):
    """Run a parameter sweep job."""
    config = ScanConfig(
        param_name=job['param_name'],
        start=float(job['start']),
        stop=float(job['stop']),
        step=float(job['step']),
        runs_per_standpoint=int(job.get('runs_per_standpoint', 50)),
        base_seed=int(job.get('seed', 42)),
    )

    def on_standpoint(sp_idx, total, pval, pt):
        bar = '#' * int(sp_idx / total * 30)
        pad = '.' * (30 - len(bar))
        print(f"\r  [{bar}{pad}] {sp_idx}/{total}  "
              f"{job['param_name']}={pval:.1f}  "
              f"I(X)={pt.avg_morans_i_x:+.4f}  I(Y)={pt.avg_morans_i_y:+.4f}", end='')

    result = run_parameter_scan(params, config, standpoint_callback=on_standpoint)
    print()

    total_sims = len(result.points) * config.runs_per_standpoint
    csv_content = generate_scan_csv(result)
    path = save_log_file(csv_content, RESULTS_DIR, 'sweep', total_sims, label=job['param_name'])

    fig = build_sweep_screenshot_figure(result)
    png = save_screenshot(fig, path)

    print(f"  CSV  -> {os.path.basename(path)}")
    if png:
        print(f"  PNG  -> {os.path.basename(png)}")
    else:
        print("  PNG  -> skipped (kaleido not available)")


def run_batch_job(job, params):
    """Run a batch simulation job."""
    n = int(job['num_runs'])

    def progress(done, total):
        bar = '#' * int(done / total * 30)
        pad = '.' * (30 - len(bar))
        print(f"\r  [{bar}{pad}] {done}/{total}", end='')

    result = run_batch_simulation(params, n, progress_callback=progress)
    print()

    csv_content = generate_batch_csv(result)
    path = save_log_file(csv_content, RESULTS_DIR, 'batch', n)

    fig = build_run_screenshot_figure(result.runs[-1], result)
    png = save_screenshot(fig, path)

    stats = result.statistics
    print(f"  Mean I(X)={stats.avg_morans_i_x:+.4f} +/- {stats.std_morans_i_x:.4f}  "
          f"I(Y)={stats.avg_morans_i_y:+.4f} +/- {stats.std_morans_i_y:.4f}")
    print(f"  CSV  -> {os.path.basename(path)}")
    if png:
        print(f"  PNG  -> {os.path.basename(png)}")


def run_convergence_job(job, params):
    """Run a convergence analysis job."""
    max_n = int(job.get('max_n', 200))
    seed  = int(job.get('seed', 42))

    def progress(done, total):
        bar = '#' * int(done / total * 30)
        pad = '.' * (30 - len(bar))
        print(f"\r  [{bar}{pad}] {done}/{total}", end='')

    points = run_convergence_analysis(params, max_n=max_n, seed=seed,
                                      progress_callback=progress)
    print()

    csv_content = generate_convergence_csv(points, params, max_n, seed)
    path = save_log_file(csv_content, RESULTS_DIR, 'convergence', max_n)

    fig = build_convergence_screenshot_figure(points, params, max_n, seed)
    png = save_screenshot(fig, path)

    last = points[-1]
    null_b = last.null_baseline
    vx = "PROVEN" if last.ci_upper_x < null_b else "NOT PROVEN"
    vy = "PROVEN" if last.ci_upper_y < null_b else "NOT PROVEN"
    print(f"  X: mean={last.mean_x:+.4f}  CI=[{last.ci_lower_x:.4f},{last.ci_upper_x:.4f}]  {vx}")
    print(f"  Y: mean={last.mean_y:+.4f}  CI=[{last.ci_lower_y:.4f},{last.ci_upper_y:.4f}]  {vy}")
    print(f"  CSV  -> {os.path.basename(path)}")
    if png:
        print(f"  PNG  -> {os.path.basename(png)}")


JOB_RUNNERS = {
    'sweep': run_sweep_job,
    'batch': run_batch_job,
    'convergence': run_convergence_job,
}


# -- CSV loader -------------------------------------------------------------

def load_matrix_csv(csv_path: str):
    """
    Load a matrix CSV file.
    Returns (defaults_dict, list_of_job_dicts).

    A row with type=defaults sets default values for all jobs.
    All other rows are jobs. Empty cells inherit from defaults.
    """
    defaults = {}
    jobs = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row_to_dict(row)
            if not d:
                continue
            if d.get('type', '').lower() == 'defaults':
                defaults = d
            else:
                jobs.append(d)

    return defaults, jobs


# -- Main -------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_matrix.py <matrix_file.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    defaults, jobs = load_matrix_csv(csv_path)

    print(f"Matrix loaded: {len(jobs)} jobs from {os.path.basename(csv_path)}")
    if defaults:
        print(f"Defaults row found ({len(defaults)} fields)")
    print(f"Results -> {RESULTS_DIR}")
    print("=" * 60)

    t_start = time.time()

    for i, job in enumerate(jobs):
        label = job.get('label', f"Job {i+1}")
        jtype = job.get('type', 'unknown')
        runner = JOB_RUNNERS.get(jtype)

        print(f"\n[{i+1}/{len(jobs)}] {label}  (type: {jtype})")
        print("-" * 50)

        if runner is None:
            print(f"  WARNING: Unknown job type '{jtype}' -- skipped")
            continue

        # Build params: SimulationParams defaults <- CSV defaults row <- job row
        merged = merge_dicts(defaults, job)
        params = params_from_dict(merged)

        try:
            runner(job, params)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"All {len(jobs)} jobs complete in {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
