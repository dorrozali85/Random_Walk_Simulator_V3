"""
CSV Export Module
Generates CSV files for single runs and batch runs per specification.
"""

import csv
import io
from typing import List
from simulation.engine import SimulationResult
from simulation.batch import BatchResult


def generate_single_run_csv(result: SimulationResult) -> str:
    """
    Generate CSV content for a single simulation run.
    
    Format:
    - Header: Timestamp,Event,PositionX,PositionY,AngleChange
    - Events: Start, WallHit, WaterSample
    - Summary section at end
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
    
    return output.getvalue()


def generate_batch_csv(batch_result: BatchResult) -> str:
    """
    Generate CSV content for batch runs.
    
    Format (long format):
    - Header: Run#,Timestamp,Event,PositionX,PositionY,AngleChange
    - All runs stacked vertically
    - Summary after each run
    - Blank line between runs
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Run#', 'Timestamp', 'Event', 'PositionX', 'PositionY', 'AngleChange'])
    
    # Write each run
    for run_idx, result in enumerate(batch_result.runs, start=1):
        # Write events for this run
        for event in result.events:
            writer.writerow([
                run_idx,
                f"{event.timestamp:.2f}",
                event.event_type,
                f"{event.position_x:.4f}",
                f"{event.position_y:.4f}",
                f"{event.angle_change:.2f}"
            ])
        
        # Write summary for this run
        writer.writerow([])
        writer.writerow([run_idx, '--- SUMMARY ---', '', '', '', ''])
        writer.writerow([run_idx, 'Moran\'s I (X)', f"{result.morans_i_x:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Moran\'s I (Y)', f"{result.morans_i_y:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Coverage %', f"{result.coverage_percent:.2f}", '', '', ''])
        writer.writerow([run_idx, 'Min Distance (m)', f"{result.min_distance:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Max Distance (m)', f"{result.max_distance:.4f}", '', '', ''])
        writer.writerow([run_idx, 'Avg Distance (m)', f"{result.avg_distance:.4f}", '', '', ''])
        writer.writerow([])  # Blank line between runs
    
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
    
    return output.getvalue()


def get_csv_filename(batch: bool, num_runs: int = 1) -> str:
    """Generate appropriate filename for CSV export."""
    if batch and num_runs > 1:
        return f"simulation_batch_{num_runs}_runs.csv"
    else:
        return "simulation_single_run.csv"
