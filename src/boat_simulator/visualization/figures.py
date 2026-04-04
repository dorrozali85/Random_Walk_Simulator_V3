# Version: v3.3  |  Date: 2026-04-04
"""
Standalone figure builders for screenshots and CLI exports.
All functions are pure Plotly/NumPy — no Streamlit dependency.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation.engine import SimulationParams, SimulationResult
from simulation.batch import BatchResult
from simulation.parameter_scan import ScanResult, SCANNABLE_PARAMS
from simulation.convergence_analysis import ConvergencePoint


def save_screenshot(fig: go.Figure, base_path: str) -> str:
    """
    Save a Plotly figure as a PNG screenshot alongside the log file.
    base_path should be the CSV path; .csv will be replaced with .png.
    Returns the saved PNG path, or None if kaleido is unavailable.
    """
    png_path = base_path.replace('.csv', '.png')
    try:
        fig.write_image(png_path, width=1400, height=900, scale=2)
        return png_path
    except Exception:
        return None


def build_run_screenshot_figure(result, batch_result=None) -> go.Figure:
    """
    Build a composite figure for a simulator run (single or batch):
    - Left subplot: boat path with sample points
    - Right subplot: stats table (Moran's I, coverage, distances) + full parameter set
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=("Simulation Path", "Run Statistics"),
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )

    # Left: path trace
    if result.path:
        px = [pt.x for pt in result.path]
        py = [pt.y for pt in result.path]
        fig.add_trace(go.Scatter(
            x=px, y=py, mode='lines', name='Path',
            line=dict(color='#3498db', width=1), opacity=0.6,
        ), row=1, col=1)

    # Sample points
    if result.samples:
        sx = [s.x for s in result.samples]
        sy = [s.y for s in result.samples]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode='markers+text', name='Samples',
            marker=dict(size=12, color='#e74c3c', symbol='circle'),
            text=[str(s.sample_number) for s in result.samples],
            textposition='top center',
        ), row=1, col=1)

    # Pool boundary
    p = result.params
    fig.add_shape(type='rect', x0=0, y0=0, x1=p.pool_width, y1=p.pool_height,
                  line=dict(color='#1e3a5f', width=2), row=1, col=1)

    # Right: stats as a table
    if batch_result is not None:
        stats = batch_result.statistics
        labels = ["Avg Moran's I (X)", "Avg Moran's I (Y)", "Avg Coverage %",
                  "Avg Min Dist (m)", "Avg Avg Dist (m)", "Avg Max Dist (m)",
                  "Runs"]
        values = [
            f"{stats.avg_morans_i_x:.4f} ± {stats.std_morans_i_x:.4f}",
            f"{stats.avg_morans_i_y:.4f} ± {stats.std_morans_i_y:.4f}",
            f"{stats.avg_coverage:.1f}% ± {stats.std_coverage:.1f}%",
            f"{stats.avg_min_distance:.2f}",
            f"{stats.avg_avg_distance:.2f}",
            f"{stats.avg_max_distance:.2f}",
            str(stats.num_runs),
        ]
    else:
        labels = ["Moran's I (X)", "Moran's I (Y)", "Coverage %",
                  "Min Dist (m)", "Avg Dist (m)", "Max Dist (m)",
                  "Wall Hits", "Total Time (min)"]
        values = [
            f"{result.morans_i_x:.4f}",
            f"{result.morans_i_y:.4f}",
            f"{result.coverage_percent:.1f}%",
            f"{result.min_distance:.2f}",
            f"{result.avg_distance:.2f}",
            f"{result.max_distance:.2f}",
            str(result.num_wall_hits),
            f"{result.total_time / 60:.1f}",
        ]

    # Append full parameter set to stats table
    labels += [
        "", "─── Run Parameters ───",
        "Pool Width (m)", "Pool Height (m)",
        "Initial Angle (°)", "Min Delta (°)", "Max Delta (°)",
        "Cruise Speed (m/s)", "Slowdown Factor", "Edge Buffer (m)",
        "Boat Width (m)", "Stop Time (s)", "Accel. (m/s²)",
        "Sample Interval (min)", "Max Samples",
    ]
    values += [
        "", "",
        str(p.pool_width), str(p.pool_height),
        f"{p.alpha}°", f"{p.min_delta}°", f"{p.max_delta}°",
        f"{p.cruise_speed} m/s", str(p.slowdown_factor), f"{p.edge_buffer} m",
        f"{p.boat_width} m", f"{p.stop_time} s", f"{p.acceleration} m/s²",
        f"{p.sample_interval} min", str(p.max_samples),
    ]
    row_colors = []
    for lbl in labels:
        if lbl == "":
            row_colors.append('#d0f0c0')
        elif lbl.startswith("─"):
            row_colors.append('#dce8f0')
        else:
            row_colors.append('#f5f7fa')

    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"],
                    fill_color='#1e3a5f', font=dict(color='white', size=13),
                    align='left'),
        cells=dict(values=[labels, values],
                   fill_color=[[c for c in row_colors], [c for c in row_colors]],
                   align='left', font=dict(size=12), height=22),
    ), row=1, col=2)

    run_type = f"Batch ({batch_result.statistics.num_runs} runs)" if batch_result else "Single Run"
    fig.update_layout(
        title=dict(text=f"Boat Simulator — {run_type}", font=dict(size=18)),
        height=700,
        plot_bgcolor='white',
        showlegend=True,
    )
    fig.update_xaxes(title_text="Width (m)", scaleanchor="y", row=1, col=1)
    fig.update_yaxes(title_text="Length (m)", row=1, col=1)
    return fig


def build_sweep_screenshot_figure(scan_result: ScanResult) -> go.Figure:
    """
    Build a composite screenshot figure for a sweep run:
    - Left column (2 rows): Moran's I chart (top) + Gradient chart (bottom)
    - Right column (spans both rows): summary results table + fixed parameters
    """
    config = scan_result.config
    param_meta = SCANNABLE_PARAMS[config.param_name]
    param_label = f"{param_meta['label']} ({param_meta['unit'].strip()})"

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.65, 0.35],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=("Moran's I vs Parameter Value", "Scan Summary", "Gradient (Rate of Change)", ""),
        specs=[
            [{"type": "xy"}, {"type": "table", "rowspan": 2}],
            [{"type": "xy"}, None],
        ],
    )

    pv = scan_result.param_values
    mi_x = scan_result.morans_i_x_values
    mi_y = scan_result.morans_i_y_values
    n_runs = scan_result.config.runs_per_standpoint
    std_x = np.array([p.std_morans_i_x for p in scan_result.points])
    std_y = np.array([p.std_morans_i_y for p in scan_result.points])
    ci_x = 1.96 * std_x / np.sqrt(max(n_runs, 1))
    ci_y = 1.96 * std_y / np.sqrt(max(n_runs, 1))
    null_baseline = -1.0 / max(scan_result.fixed_params.max_samples - 1, 1)

    # Top-left: Moran's I lines with 95% CI
    fig.add_trace(go.Scatter(
        x=pv, y=mi_x, mode='lines+markers', name="Moran's I (X) ±95% CI",
        line=dict(color='#e74c3c', width=2), marker=dict(size=6),
        error_y=dict(type='data', array=ci_x, visible=True, color='rgba(231,76,60,0.3)'),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pv, y=mi_y, mode='lines+markers', name="Moran's I (Y) ±95% CI",
        line=dict(color='#3498db', width=2), marker=dict(size=6),
        error_y=dict(type='data', array=ci_y, visible=True, color='rgba(52,152,219,0.3)'),
    ), row=1, col=1)
    fig.add_hline(y=null_baseline, line_dash="dot", line_color="#888",
                  annotation_text=f"Null E[I]={null_baseline:.3f}",
                  annotation_font_color="#888", row=1, col=1)

    # Bottom-left: Gradient
    if scan_result.gradient_x is not None:
        fig.add_trace(go.Scatter(
            x=pv, y=np.abs(scan_result.gradient_x), mode='lines+markers',
            name='|Gradient X|', line=dict(color='#e74c3c', width=2, dash='dot'), marker=dict(size=5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=pv, y=np.abs(scan_result.gradient_y), mode='lines+markers',
            name='|Gradient Y|', line=dict(color='#3498db', width=2, dash='dot'), marker=dict(size=5),
        ), row=2, col=1)
        peak_grad = max(np.max(np.abs(scan_result.gradient_x)), np.max(np.abs(scan_result.gradient_y)))
        fig.add_hline(y=0.1 * peak_grad, line_dash="dash", line_color="gray",
                      annotation_text="10% threshold", row=2, col=1)

    # Convergence + best markers on both charts
    for idx, color, label in [
        (scan_result.convergence_index_x, '#e74c3c', 'Convergence X'),
        (scan_result.convergence_index_y, '#3498db', 'Convergence Y'),
    ]:
        if idx is not None:
            cv_val = pv[idx]
            fig.add_vline(x=cv_val, line_dash="dash", line_color=color,
                          annotation_text=f"{label}: {cv_val:.1f}", row=1, col=1)
            fig.add_vline(x=cv_val, line_dash="dash", line_color=color, row=2, col=1)

    if scan_result.best_combined_index is not None:
        bi = scan_result.best_combined_index
        best_pv = pv[bi]
        fig.add_vline(x=best_pv, line_dash="solid", line_color="#2ecc71", line_width=2,
                      annotation_text=f"Best: {best_pv:.1f}", annotation_font_color="#2ecc71",
                      row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[best_pv, best_pv], y=[mi_x[bi], mi_y[bi]],
            mode='markers', name='Best (lowest combined)',
            marker=dict(size=16, color='#2ecc71', symbol='star',
                        line=dict(width=1, color='darkgreen')),
        ), row=1, col=1)

    # Right column: summary table
    best_pt = scan_result.points[scan_result.best_combined_index] if scan_result.best_combined_index is not None else None
    cv_x_val = f"{pv[scan_result.convergence_index_x]:.1f}{param_meta['unit']}" if scan_result.convergence_index_x is not None else "Not found"
    cv_y_val = f"{pv[scan_result.convergence_index_y]:.1f}{param_meta['unit']}" if scan_result.convergence_index_y is not None else "Not found"

    table_labels = [
        "Scanned Parameter",
        "Scan Range",
        "Step",
        "Runs / Standpoint",
        "Total Standpoints",
        "Total Simulations",
        "",
        "Best Parameter Value",
        "Best Combined Score",
        "Best Moran's I (X)",
        "Best Moran's I (Y)",
        "Best Coverage %",
        "",
        "Convergence (X)",
        "Convergence (Y)",
    ]
    table_values = [
        param_meta['label'],
        f"{config.start}{param_meta['unit']} → {config.stop}{param_meta['unit']}",
        f"{config.step}{param_meta['unit']}",
        str(config.runs_per_standpoint),
        str(len(scan_result.points)),
        str(len(scan_result.points) * config.runs_per_standpoint),
        "",
        f"{best_pt.param_value:.1f}{param_meta['unit']}" if best_pt else "N/A",
        f"{best_pt.avg_morans_i_x + best_pt.avg_morans_i_y:.4f}" if best_pt else "N/A",
        f"{best_pt.avg_morans_i_x:.4f}" if best_pt else "N/A",
        f"{best_pt.avg_morans_i_y:.4f}" if best_pt else "N/A",
        f"{best_pt.avg_coverage:.1f}%" if best_pt else "N/A",
        "",
        cv_x_val,
        cv_y_val,
    ]

    # Append fixed parameter set to summary table
    fp = scan_result.fixed_params
    table_labels += [
        "", "─── Fixed Parameters ───",
        "Pool Width (m)", "Pool Height (m)",
        "Initial Angle (°)", "Min Delta (°)", "Max Delta (°)",
        "Cruise Speed (m/s)", "Slowdown Factor", "Edge Buffer (m)",
        "Boat Width (m)", "Stop Time (s)", "Accel. (m/s²)",
        "Sample Interval (min)", "Max Samples",
    ]
    table_values += [
        "", "",
        str(fp.pool_width), str(fp.pool_height),
        f"{fp.alpha}°", f"{fp.min_delta}°", f"{fp.max_delta}°",
        f"{fp.cruise_speed} m/s", str(fp.slowdown_factor), f"{fp.edge_buffer} m",
        f"{fp.boat_width} m", f"{fp.stop_time} s", f"{fp.acceleration} m/s²",
        f"{fp.sample_interval} min", str(fp.max_samples),
    ]

    row_colors = []
    for lbl in table_labels:
        if lbl == "":
            row_colors.append('#d0f0c0')
        elif lbl.startswith("─"):
            row_colors.append('#dce8f0')
        elif lbl.startswith("Best"):
            row_colors.append('#e8f8e8')
        elif lbl.startswith("Convergence"):
            row_colors.append('#e8f0f8')
        else:
            row_colors.append('#f5f7fa')

    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"],
                    fill_color='#1e3a5f', font=dict(color='white', size=13), align='left'),
        cells=dict(
            values=[table_labels, table_values],
            fill_color=[[c for c in row_colors], [c for c in row_colors]],
            align='left', font=dict(size=12),
            height=24,
        ),
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"Parameter Sweep — {param_meta['label']}", font=dict(size=18)),
        height=850,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.3),
    )
    fig.update_xaxes(title_text=param_label, row=1, col=1)
    fig.update_xaxes(title_text=param_label, row=2, col=1)
    fig.update_yaxes(title_text="Moran's I", row=1, col=1)
    fig.update_yaxes(title_text="|Gradient|", row=2, col=1)
    return fig


def build_convergence_screenshot_figure(points, params, max_n: int, seed: int) -> go.Figure:
    """
    Build composite screenshot figure for a convergence analysis run:
    - Left column (2 rows): Mean ± 95% CI chart (top) + SE chart (bottom)
    - Right column (spans both rows): analysis config + final results + full parameters table
    """
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.65, 0.35],
        row_heights=[0.60, 0.40],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Mean Moran's I ± 95% Confidence Interval vs N",
            "Analysis Config & Parameters",
            "Standard Error vs N  (narrows as 1/√N)",
            "",
        ),
        specs=[
            [{"type": "xy"}, {"type": "table", "rowspan": 2}],
            [{"type": "xy"}, None],
        ],
    )

    if not points:
        return fig

    ns        = [pt.n for pt in points]
    null_b    = points[0].null_baseline

    for axis, color, name in [
        ('x', '#e74c3c', "Moran's I (X)"),
        ('y', '#3498db', "Moran's I (Y)"),
    ]:
        means     = [getattr(pt, f'mean_{axis}')     for pt in points]
        ci_lowers = [getattr(pt, f'ci_lower_{axis}') for pt in points]
        ci_uppers = [getattr(pt, f'ci_upper_{axis}') for pt in points]
        ses       = [getattr(pt, f'se_{axis}')       for pt in points]

        # CI shaded band (top chart)
        fig.add_trace(go.Scatter(
            x=ns + ns[::-1], y=ci_uppers + ci_lowers[::-1],
            fill='toself', fillcolor=color, opacity=0.15,
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ), row=1, col=1)
        # Mean line (top chart)
        fig.add_trace(go.Scatter(
            x=ns, y=means, mode='lines+markers',
            name=name, line=dict(color=color, width=2), marker=dict(size=5),
        ), row=1, col=1)
        # SE line (bottom chart)
        fig.add_trace(go.Scatter(
            x=ns, y=ses, mode='lines+markers',
            name=f'SE ({axis.upper()})', line=dict(color=color, width=2, dash='dot'),
            marker=dict(size=4),
        ), row=2, col=1)

    # Null baseline
    fig.add_hline(y=null_b, line_dash="dash", line_color="#e67e22", line_width=1.5,
                  annotation_text=f"Null E[I] = {null_b:.3f}",
                  annotation_font_color="#e67e22", row=1, col=1)

    # ── Parameters + summary table ─────────────────────────────────────────
    last = points[-1]
    verdict_x = "PROVEN"     if last.ci_upper_x < null_b else "NOT PROVEN"
    verdict_y = "PROVEN"     if last.ci_upper_y < null_b else "NOT PROVEN"

    tbl_labels = [
        "─── Analysis Config ───",
        "Total Runs (N)", "Random Seed",
        "Null Baseline E[I]", "Checkpoints",
        "",
        f"─── Final Results (N={last.n}) ───",
        "Mean I(X)", "95% CI (X)", "SE (X)",
        "Mean I(Y)", "95% CI (Y)", "SE (Y)",
        "Verdict (X)", "Verdict (Y)",
        "",
        "─── Fixed Parameters ───",
        "Pool Width (m)", "Pool Height (m)",
        "Initial Angle (°)", "Min Delta (°)", "Max Delta (°)",
        "Cruise Speed (m/s)", "Slowdown Factor", "Edge Buffer (m)",
        "Boat Width (m)", "Stop Time (s)", "Accel. (m/s²)",
        "Sample Interval (min)", "Max Samples",
    ]
    tbl_values = [
        "",
        str(max_n), str(seed),
        f"{null_b:.4f}", str(len(points)),
        "",
        "",
        f"{last.mean_x:.4f}",
        f"[{last.ci_lower_x:.4f}, {last.ci_upper_x:.4f}]",
        f"{last.se_x:.4f}",
        f"{last.mean_y:.4f}",
        f"[{last.ci_lower_y:.4f}, {last.ci_upper_y:.4f}]",
        f"{last.se_y:.4f}",
        verdict_x, verdict_y,
        "",
        "",
        str(params.pool_width), str(params.pool_height),
        f"{params.alpha}°", f"{params.min_delta}°", f"{params.max_delta}°",
        f"{params.cruise_speed} m/s", str(params.slowdown_factor), f"{params.edge_buffer} m",
        f"{params.boat_width} m", f"{params.stop_time} s", f"{params.acceleration} m/s²",
        f"{params.sample_interval} min", str(params.max_samples),
    ]

    row_colors_lbl = []
    row_colors_val = []
    for i, lbl in enumerate(tbl_labels):
        val = tbl_values[i]
        if lbl == "":
            row_colors_lbl.append('#d0f0c0')
            row_colors_val.append('#d0f0c0')
        elif lbl.startswith("─"):
            row_colors_lbl.append('#dce8f0')
            row_colors_val.append('#dce8f0')
        elif lbl == "Verdict (X)":
            row_colors_lbl.append('#f5f7fa')
            row_colors_val.append('#c8f0c8' if val == "PROVEN" else '#fde8e8')
        elif lbl == "Verdict (Y)":
            row_colors_lbl.append('#f5f7fa')
            row_colors_val.append('#c8f0c8' if val == "PROVEN" else '#fde8e8')
        else:
            row_colors_lbl.append('#f5f7fa')
            row_colors_val.append('#f5f7fa')

    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"],
                    fill_color='#1e3a5f', font=dict(color='white', size=13), align='left'),
        cells=dict(
            values=[tbl_labels, tbl_values],
            fill_color=[row_colors_lbl, row_colors_val],
            align='left', font=dict(size=12), height=24,
        ),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Number of Runs (N)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Runs (N)", row=2, col=1)
    fig.update_yaxes(title_text="Moran's I", row=1, col=1)
    fig.update_yaxes(title_text="Standard Error", row=2, col=1)
    fig.update_layout(
        title=dict(text="Convergence Analysis — Full Report", font=dict(size=18)),
        height=900,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.3),
    )
    return fig
