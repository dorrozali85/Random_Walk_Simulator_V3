# Version: v3.2  |  Date: 2026-04-03
"""
Boat Simulator - Streamlit Application
A simulation tool for modeling a robotic boat performing correlated random walk
in a rectangular pool, with water sampling and statistical analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import os
from datetime import datetime

# Import simulation modules
from simulation.engine import SimulationParams, run_single_simulation
from simulation.statistics import calculate_all_statistics
from simulation.batch import run_batch_simulation
from simulation.parameter_scan import (
    ScanConfig, ScanResult, SCANNABLE_PARAMS, run_parameter_scan
)
from visualization.plotting import create_path_figure, create_animated_figure, create_coverage_heatmap
from export.csv_logger import generate_single_run_csv, generate_batch_csv, generate_scan_csv, get_csv_filename, save_log_file
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Results directory — module-level so all functions can access it
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Boat Simulator",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3a5f;
    }
    .bulk-stats {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e9f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #3498db;
    }
    .log-area {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'simulation_result': None,
        'batch_result': None,
        'scan_result': None,
        'animation_progress': 1.0,
        'analysis_mode': False,
        'show_animation': False,
        'run_count': 0,
        'app_mode': 'simulator',
        'last_saved_path': None,
        'boat_width': 0.6,
        'stop_time': 2.0,
        'acceleration': 0.1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_params_from_inputs() -> SimulationParams:
    """Create SimulationParams from sidebar inputs."""
    return SimulationParams(
        pool_width=st.session_state.pool_width,
        pool_height=st.session_state.pool_height,
        alpha=st.session_state.alpha,
        min_delta=st.session_state.min_delta,
        max_delta=st.session_state.max_delta,
        sample_interval=st.session_state.sample_interval,
        max_samples=st.session_state.max_samples,
        cruise_speed=st.session_state.cruise_speed,
        slowdown_factor=st.session_state.slowdown_factor,
        edge_buffer=st.session_state.edge_buffer,
        boat_width=st.session_state.boat_width,
        stop_time=st.session_state.stop_time,
        acceleration=st.session_state.acceleration,
    )


def display_single_run_stats(result):
    """Display statistics for a single run."""
    st.markdown('<p class="section-header">📊 Single Run Statistics</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Moran's I (X)</div>
            <div class="stat-value">{result.morans_i_x:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Min Distance</div>
            <div class="stat-value">{result.min_distance:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Moran's I (Y)</div>
            <div class="stat-value">{result.morans_i_y:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Avg Distance</div>
            <div class="stat-value">{result.avg_distance:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Coverage</div>
            <div class="stat-value">{result.coverage_percent:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Max Distance</div>
            <div class="stat-value">{result.max_distance:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Points", len(result.samples))
    col2.metric("Wall Hits", result.num_wall_hits)
    col3.metric("Lag-1 Corr", f"{result.lag1_correlation:.4f}")
    col4.metric("Total Time", f"{result.total_time/60:.1f} min")


def display_bulk_stats(batch_result):
    """Display aggregated statistics from batch runs."""
    stats = batch_result.statistics
    
    st.markdown(f"""
    <div class="bulk-stats">
        <h3 style="color: #1e3a5f; margin-top: 0;">📈 Batch Statistics ({stats.num_runs} runs)</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px;"><strong>Avg Moran's I (X)</strong></td>
                <td style="padding: 8px;">{stats.avg_morans_i_x:.4f} ± {stats.std_morans_i_x:.4f}</td>
            </tr>
            <tr style="background: rgba(255,255,255,0.5);">
                <td style="padding: 8px;"><strong>Avg Moran's I (Y)</strong></td>
                <td style="padding: 8px;">{stats.avg_morans_i_y:.4f} ± {stats.std_morans_i_y:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><strong>Avg Coverage</strong></td>
                <td style="padding: 8px;">{stats.avg_coverage:.1f}% ± {stats.std_coverage:.1f}%</td>
            </tr>
            <tr style="background: rgba(255,255,255,0.5);">
                <td style="padding: 8px;"><strong>Avg Min Distance</strong></td>
                <td style="padding: 8px;">{stats.avg_min_distance:.2f} m</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><strong>Avg Avg Distance</strong></td>
                <td style="padding: 8px;">{stats.avg_avg_distance:.2f} m</td>
            </tr>
            <tr style="background: rgba(255,255,255,0.5);">
                <td style="padding: 8px;"><strong>Avg Max Distance</strong></td>
                <td style="padding: 8px;">{stats.avg_max_distance:.2f} m</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


def display_event_log(result, max_events=50):
    """Display scrollable event log."""
    st.markdown('<p class="section-header">📜 Event Log</p>', unsafe_allow_html=True)
    
    log_text = ""
    for event in result.events[:max_events]:
        time_str = f"{event.timestamp/60:.2f} min"
        pos_str = f"({event.position_x:.2f}, {event.position_y:.2f})"
        
        if event.event_type == 'Start':
            log_text += f"[{time_str}] 🚀 START at {pos_str}, angle={event.angle_change:.1f}°\n"
        elif event.event_type == 'WallHit':
            log_text += f"[{time_str}] 💥 WALL HIT at {pos_str}, Δangle={event.angle_change:+.1f}°\n"
        elif event.event_type == 'WaterSample':
            log_text += f"[{time_str}] 💧 SAMPLE at {pos_str}\n"
    
    if len(result.events) > max_events:
        log_text += f"\n... and {len(result.events) - max_events} more events"
    
    st.markdown(f'<div class="log-area"><pre>{log_text}</pre></div>', unsafe_allow_html=True)


def save_screenshot(fig: go.Figure, base_path: str) -> str:
    """
    Save a Plotly figure as a PNG screenshot alongside the log file.
    base_path should be the CSV path (without .csv); .png will be appended.
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
    - Right subplot: stats table (Moran's I, coverage, distances)
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

    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"],
                    fill_color='#1e3a5f', font=dict(color='white', size=13),
                    align='left'),
        cells=dict(values=[labels, values],
                   fill_color=[['#f5f7fa', '#e4e8ec'] * (len(labels) // 2 + 1)],
                   align='left', font=dict(size=12)),
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
    - Right column (spans both rows): summary results table
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
    std_x = np.array([p.std_morans_i_x for p in scan_result.points])
    std_y = np.array([p.std_morans_i_y for p in scan_result.points])

    # Top-left: Moran's I lines
    fig.add_trace(go.Scatter(
        x=pv, y=mi_x, mode='lines+markers', name="Moran's I (X)",
        line=dict(color='#e74c3c', width=2), marker=dict(size=6),
        error_y=dict(type='data', array=std_x, visible=True, color='rgba(231,76,60,0.3)'),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pv, y=mi_y, mode='lines+markers', name="Moran's I (Y)",
        line=dict(color='#3498db', width=2), marker=dict(size=6),
        error_y=dict(type='data', array=std_y, visible=True, color='rgba(52,152,219,0.3)'),
    ), row=1, col=1)

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

    row_colors = []
    for lbl in table_labels:
        if lbl == "":
            row_colors.append('#d0f0c0')  # light green separator
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


def create_scan_results_figure(scan_result: ScanResult) -> go.Figure:
    """Create a two-subplot figure showing Moran's I and gradient vs parameter value."""
    config = scan_result.config
    param_meta = SCANNABLE_PARAMS[config.param_name]
    param_label = f"{param_meta['label']} ({param_meta['unit'].strip()})"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Moran's I vs Parameter Value", "Gradient (Rate of Change)"),
        vertical_spacing=0.15,
        row_heights=[0.55, 0.45],
    )

    pv = scan_result.param_values
    mi_x = scan_result.morans_i_x_values
    mi_y = scan_result.morans_i_y_values
    std_x = np.array([p.std_morans_i_x for p in scan_result.points])
    std_y = np.array([p.std_morans_i_y for p in scan_result.points])

    # Top plot: Moran's I with error bands
    fig.add_trace(go.Scatter(
        x=pv, y=mi_x, mode='lines+markers', name="Moran's I (X)",
        line=dict(color='#e74c3c', width=2), marker=dict(size=6),
        error_y=dict(type='data', array=std_x, visible=True, color='rgba(231,76,60,0.3)'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pv, y=mi_y, mode='lines+markers', name="Moran's I (Y)",
        line=dict(color='#3498db', width=2), marker=dict(size=6),
        error_y=dict(type='data', array=std_y, visible=True, color='rgba(52,152,219,0.3)'),
    ), row=1, col=1)

    # Bottom plot: Gradient
    if scan_result.gradient_x is not None:
        fig.add_trace(go.Scatter(
            x=pv, y=np.abs(scan_result.gradient_x), mode='lines+markers',
            name='|Gradient X|', line=dict(color='#e74c3c', width=2, dash='dot'),
            marker=dict(size=5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=pv, y=np.abs(scan_result.gradient_y), mode='lines+markers',
            name='|Gradient Y|', line=dict(color='#3498db', width=2, dash='dot'),
            marker=dict(size=5),
        ), row=2, col=1)

        # Threshold line
        peak_grad = max(np.max(np.abs(scan_result.gradient_x)),
                        np.max(np.abs(scan_result.gradient_y)))
        threshold = 0.1 * peak_grad
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                      annotation_text="10% threshold", row=2, col=1)

    # Convergence markers
    for idx, color, label in [
        (scan_result.convergence_index_x, '#e74c3c', 'Convergence X'),
        (scan_result.convergence_index_y, '#3498db', 'Convergence Y'),
    ]:
        if idx is not None:
            cv_val = pv[idx]
            fig.add_vline(x=cv_val, line_dash="dash", line_color=color,
                          annotation_text=f"{label}: {cv_val:.1f}", row=1, col=1)
            fig.add_vline(x=cv_val, line_dash="dash", line_color=color, row=2, col=1)

    # Best combined point marker (green star)
    if scan_result.best_combined_index is not None:
        bi = scan_result.best_combined_index
        best_pv = pv[bi]
        fig.add_vline(x=best_pv, line_dash="solid", line_color="#2ecc71", line_width=2,
                      annotation_text=f"Best: {best_pv:.1f}",
                      annotation_font_color="#2ecc71", row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[best_pv, best_pv], y=[mi_x[bi], mi_y[bi]],
            mode='markers', name='Best (lowest combined)',
            marker=dict(size=16, color='#2ecc71', symbol='star',
                        line=dict(width=1, color='darkgreen')),
            showlegend=True,
        ), row=1, col=1)

    fig.update_layout(
        height=650,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    fig.update_xaxes(title_text=param_label, row=2, col=1)
    fig.update_xaxes(title_text=param_label, row=1, col=1)
    fig.update_yaxes(title_text="Moran's I", row=1, col=1)
    fig.update_yaxes(title_text="|Gradient|", row=2, col=1)

    return fig


def create_live_scan_figure(points, param_name, param_values_so_far):
    """Create a live-updating figure during scan progress."""
    param_meta = SCANNABLE_PARAMS[param_name]
    param_label = f"{param_meta['label']} ({param_meta['unit'].strip()})"

    fig = go.Figure()
    pv = [p.param_value for p in points]
    mi_x = [p.avg_morans_i_x for p in points]
    mi_y = [p.avg_morans_i_y for p in points]
    std_x = [p.std_morans_i_x for p in points]
    std_y = [p.std_morans_i_y for p in points]

    fig.add_trace(go.Scatter(
        x=pv, y=mi_x, mode='lines+markers', name="Moran's I (X)",
        line=dict(color='#e74c3c', width=2), marker=dict(size=7),
        error_y=dict(type='data', array=std_x, visible=True, color='rgba(231,76,60,0.3)'),
    ))
    fig.add_trace(go.Scatter(
        x=pv, y=mi_y, mode='lines+markers', name="Moran's I (Y)",
        line=dict(color='#3498db', width=2), marker=dict(size=7),
        error_y=dict(type='data', array=std_y, visible=True, color='rgba(52,152,219,0.3)'),
    ))

    # Mark current best point (lowest combined)
    if len(points) >= 1:
        combined = [p.avg_morans_i_x + p.avg_morans_i_y for p in points]
        best_idx = int(np.argmin(combined))
        best_p = points[best_idx]
        fig.add_trace(go.Scatter(
            x=[best_p.param_value, best_p.param_value],
            y=[best_p.avg_morans_i_x, best_p.avg_morans_i_y],
            mode='markers', name=f'Best so far ({best_p.param_value:.1f})',
            marker=dict(size=14, color='#2ecc71', symbol='star',
                        line=dict(width=1, color='darkgreen')),
        ))

    fig.update_layout(
        title="Live Scan Progress",
        xaxis_title=param_label,
        yaxis_title="Moran's I (avg)",
        height=400,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    return fig


def run_scanner_page():
    """Render the Parameter Scanner page."""
    st.markdown('<p class="section-header">Parameter Optimization Scanner</p>', unsafe_allow_html=True)
    st.markdown(
        "Sweep a single parameter across a range to find where **Moran's I stabilizes**. "
        "All other parameters stay fixed at the values set in the sidebar."
    )

    # Scan configuration
    scan_col1, scan_col2 = st.columns(2)

    with scan_col1:
        param_name = st.selectbox(
            "Parameter to Scan",
            options=list(SCANNABLE_PARAMS.keys()),
            format_func=lambda k: SCANNABLE_PARAMS[k]['label'],
            key="scan_param_name",
        )
        meta = SCANNABLE_PARAMS[param_name]

        start_val = st.number_input(
            f"Start ({meta['unit'].strip()}) (default 5)", min_value=meta['min'], max_value=meta['max'],
            value=meta.get('default_start', 5.0), step=meta['default_step'], key="scan_start",
        )
        stop_val = st.number_input(
            f"Stop ({meta['unit'].strip()}) (default 20)", min_value=meta['min'], max_value=meta['max'],
            value=meta.get('default_stop', 20.0), step=meta['default_step'], key="scan_stop",
        )

    with scan_col2:
        step_val = st.number_input(
            f"Step ({meta['unit'].strip()})", min_value=meta['default_step'] / 5,
            max_value=(meta['max'] - meta['min']) / 2,
            value=meta['default_step'], step=meta['default_step'] / 5,
            key="scan_step",
        )
        runs_per = st.number_input(
            "Runs per Standpoint (default 10)", min_value=5, max_value=1000, value=10, step=5,
            key="scan_runs_per_standpoint",
            help="Number of simulations at each parameter value for statistical confidence",
        )
        total_standpoints = len(np.arange(start_val, stop_val + step_val * 0.5, step_val))
        total_sims = total_standpoints * runs_per
        st.info(f"**{total_standpoints}** standpoints x **{runs_per}** runs = **{total_sims}** total simulations")

    st.markdown("---")

    # Run scan button
    if st.button("Start Parameter Scan", type="primary", use_container_width=True, key="run_scan_btn"):
        params = create_params_from_inputs()
        config = ScanConfig(
            param_name=param_name,
            start=start_val,
            stop=stop_val,
            step=step_val,
            runs_per_standpoint=runs_per,
            base_seed=42,
        )

        # Progress UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        live_chart = st.empty()
        live_points = []

        def on_standpoint_done(sp_idx, total_sp, param_val, point_result):
            progress_bar.progress(sp_idx / total_sp)
            unit = meta['unit']
            status_text.markdown(
                f"**Standpoint {sp_idx}/{total_sp}** completed "
                f"| {meta['label']} = **{param_val:.1f}{unit}** "
                f"| Moran's I(X) = **{point_result.avg_morans_i_x:.4f}** "
                f"| Moran's I(Y) = **{point_result.avg_morans_i_y:.4f}**"
            )
            live_points.append(point_result)
            if len(live_points) >= 2:
                live_chart.plotly_chart(
                    create_live_scan_figure(live_points, param_name, None),
                    use_container_width=True,
                )

        def on_run_done(sp_idx, total_sp, run_idx, total_runs):
            detail_text.text(
                f"Standpoint {sp_idx}/{total_sp} - Run {run_idx}/{total_runs}"
            )

        scan_result = run_parameter_scan(
            params, config,
            standpoint_callback=on_standpoint_done,
            run_callback=on_run_done,
        )

        st.session_state.scan_result = scan_result

        progress_bar.empty()
        status_text.empty()
        detail_text.empty()
        live_chart.empty()

        total_sims = len(scan_result.points) * config.runs_per_standpoint
        csv_content = generate_scan_csv(scan_result)
        saved_path = save_log_file(csv_content, RESULTS_DIR, 'sweep', total_sims)
        st.session_state.last_saved_path = saved_path
        fig_screenshot = build_sweep_screenshot_figure(scan_result)
        save_screenshot(fig_screenshot, saved_path)

        fname = os.path.basename(saved_path)
        st.success(f"Scan complete! Log saved: `{fname}`")
        st.rerun()

    # Display results if available
    if st.session_state.scan_result is not None:
        scan_result = st.session_state.scan_result
        cfg = scan_result.config
        meta_r = SCANNABLE_PARAMS[cfg.param_name]

        st.markdown("---")
        st.markdown(f'<p class="section-header">Scan Results: {meta_r["label"]}</p>', unsafe_allow_html=True)

        # Summary boxes: Convergence X, Convergence Y, Best Parameter
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            if scan_result.convergence_index_x is not None:
                cv = scan_result.points[scan_result.convergence_index_x].param_value
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Convergence (Moran's I X)</div>
                    <div class="stat-value">{cv:.1f}{meta_r['unit']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-label">Convergence (Moran's I X)</div>
                    <div class="stat-value">Not found in range</div>
                </div>
                """, unsafe_allow_html=True)

        with sum_col2:
            if scan_result.convergence_index_y is not None:
                cv = scan_result.points[scan_result.convergence_index_y].param_value
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Convergence (Moran's I Y)</div>
                    <div class="stat-value">{cv:.1f}{meta_r['unit']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-label">Convergence (Moran's I Y)</div>
                    <div class="stat-value">Not found in range</div>
                </div>
                """, unsafe_allow_html=True)

        with sum_col3:
            if scan_result.best_combined_index is not None:
                best_pt = scan_result.points[scan_result.best_combined_index]
                best_combined = best_pt.avg_morans_i_x + best_pt.avg_morans_i_y
                st.markdown(f"""
                <div class="stat-box" style="border-left: 4px solid #2ecc71;">
                    <div class="stat-label">Best Parameter (Lowest Combined Moran's I)</div>
                    <div class="stat-value" style="color: #27ae60;">{meta_r['label']} = {best_pt.param_value:.1f}{meta_r['unit']}</div>
                    <div style="font-size: 0.85rem; color: #555; margin-top: 0.3rem;">
                        Combined: <strong>{best_combined:.4f}</strong><br>
                        X: {best_pt.avg_morans_i_x:.4f} | Y: {best_pt.avg_morans_i_y:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-box" style="border-left: 4px solid #2ecc71;">
                    <div class="stat-label">Best Parameter</div>
                    <div class="stat-value">No data</div>
                </div>
                """, unsafe_allow_html=True)

        # Full results chart
        fig = create_scan_results_figure(scan_result)
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("View Raw Data Table"):
            import pandas as pd
            rows = []
            for i, pt in enumerate(scan_result.points):
                combined = pt.avg_morans_i_x + pt.avg_morans_i_y
                row = {
                    f"{meta_r['label']}": pt.param_value,
                    "Avg Moran's I (X)": f"{pt.avg_morans_i_x:.4f}",
                    "Avg Moran's I (Y)": f"{pt.avg_morans_i_y:.4f}",
                    "Combined Score": f"{combined:.4f}",
                    "Std (X)": f"{pt.std_morans_i_x:.4f}",
                    "Std (Y)": f"{pt.std_morans_i_y:.4f}",
                    "Avg Coverage %": f"{pt.avg_coverage:.1f}",
                    "Runs": pt.num_runs,
                }
                if scan_result.gradient_x is not None:
                    row["|Grad X|"] = f"{abs(scan_result.gradient_x[i]):.6f}"
                    row["|Grad Y|"] = f"{abs(scan_result.gradient_y[i]):.6f}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Export
        scan_csv = generate_scan_csv(scan_result)
        st.download_button(
            label="Export Scan Results CSV",
            data=scan_csv,
            file_name=f"scan_{cfg.param_name}_{cfg.start}_{cfg.stop}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">Boat Random Walk Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Correlated random walk simulation with spatial statistics analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Mode selector tabs
    tab_sim, tab_scan = st.tabs(["Simulator", "Parameter Scanner"])

    # Sidebar - Parameters (shared by both modes)
    with st.sidebar:
        st.header("Simulation Parameters")

        st.subheader("Pool Dimensions")
        st.number_input("Pool Width (m) (default 12.5)", min_value=5.0, max_value=1000.0, value=12.5,
                       step=0.5, key="pool_width")
        st.number_input("Pool Height (m) (default 25)", min_value=5.0, max_value=1000.0, value=25.0,
                       step=0.5, key="pool_height")

        st.subheader("Movement Parameters")
        st.number_input("Initial Angle (deg) (default 45)", min_value=0.0, max_value=360.0, value=45.0,
                       step=5.0, key="alpha")
        st.number_input("Min Delta (deg) (default 20)", min_value=0.0, max_value=270.0, value=20.0,
                       step=5.0, key="min_delta")
        st.number_input("Max Delta (deg) (default 45)", min_value=0.0, max_value=270.0, value=45.0,
                       step=5.0, key="max_delta")
        st.number_input("Cruise Speed (m/s) (default 0.2)", min_value=0.1, max_value=10.0, value=0.2,
                       step=0.05, key="cruise_speed")

        st.subheader("Edge Behavior")
        st.number_input("Slowdown Factor (default 0.5)", min_value=0.1, max_value=1.0, value=0.5,
                       step=0.1, key="slowdown_factor")
        st.number_input("Edge Buffer (m) (default 0.5)", min_value=0.1, max_value=3.0, value=0.5,
                       step=0.1, key="edge_buffer")

        st.subheader("Wall Behaviour")
        st.number_input("Boat Width (m) (default 0.6)", min_value=0.1, max_value=2.0, value=0.6,
                       step=0.1, key="boat_width")
        st.number_input("Stop Time (s) (default 2.0)", min_value=0.0, max_value=10.0, value=2.0,
                       step=0.5, key="stop_time")
        st.number_input("Acceleration (m/s²) (default 0.1)", min_value=0.01, max_value=2.0, value=0.1,
                       step=0.01, key="acceleration")

        st.subheader("Sampling")
        st.number_input("Sample Interval (min) (default 5)", min_value=1.0, max_value=60.0, value=5.0,
                       step=1.0, key="sample_interval")
        st.number_input("Max Samples (default 5)", min_value=1, max_value=100, value=5,
                       step=1, key="max_samples")

        st.markdown("---")

        st.subheader("Batch Settings")
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=2000, value=1,
                                   step=1, key="num_runs")

        # Run button
        if st.button("Run Simulation", type="primary", use_container_width=True):
            params = create_params_from_inputs()

            if num_runs == 1:
                # Single run
                with st.spinner("Running simulation..."):
                    result = run_single_simulation(params)
                    result = calculate_all_statistics(result)
                    st.session_state.simulation_result = result
                    st.session_state.batch_result = None
                    st.session_state.run_count += 1
                    csv_content = generate_single_run_csv(result)
                    saved_path = save_log_file(csv_content, RESULTS_DIR, 'single', 1)
                    st.session_state.last_saved_path = saved_path
                    fig_screenshot = build_run_screenshot_figure(result)
                    save_screenshot(fig_screenshot, saved_path)
            else:
                # Batch run
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"Running simulation {current}/{total}...")

                batch_result = run_batch_simulation(params, num_runs,
                                                    progress_callback=update_progress)

                st.session_state.batch_result = batch_result
                st.session_state.simulation_result = batch_result.runs[-1]
                st.session_state.run_count += 1

                progress_bar.empty()
                status_text.empty()

                csv_content = generate_batch_csv(batch_result)
                saved_path = save_log_file(csv_content, RESULTS_DIR, 'batch', num_runs)
                st.session_state.last_saved_path = saved_path
                fig_screenshot = build_run_screenshot_figure(batch_result.runs[-1], batch_result)
                save_screenshot(fig_screenshot, saved_path)

            fname = os.path.basename(st.session_state.last_saved_path)
            st.success(f"Simulation complete! Log saved: `{fname}`")

        # Reset button
        if st.button("Reset", use_container_width=True):
            st.session_state.simulation_result = None
            st.session_state.batch_result = None
            st.session_state.scan_result = None
            st.session_state.animation_progress = 1.0
            st.session_state.analysis_mode = False
            st.rerun()

    # === SIMULATOR TAB ===
    with tab_sim:
        # Main content area - Three columns
        if st.session_state.batch_result is not None:
            left_col, center_col, right_col = st.columns([1.2, 1.5, 1.3])
            with left_col:
                display_bulk_stats(st.session_state.batch_result)
        else:
            center_col, right_col = st.columns([1.5, 1.5])

        result = st.session_state.simulation_result

        # Center column - Visualization
        with center_col:
            st.markdown('<p class="section-header">Visualization</p>', unsafe_allow_html=True)

            if result is not None:
                view_col1, view_col2, view_col3 = st.columns(3)

                with view_col1:
                    analysis_mode = st.checkbox("Analysis Mode", value=st.session_state.analysis_mode,
                                               help="Show only samples with nearest-neighbor connections")
                    st.session_state.analysis_mode = analysis_mode

                with view_col2:
                    show_heatmap = st.checkbox("Coverage Heatmap", value=False,
                                              help="Show coverage intensity heatmap")

                with view_col3:
                    animation_progress = st.slider("Animation Progress", 0.0, 1.0, 1.0, 0.01,
                                                  help="Scrub through simulation timeline")

                if show_heatmap:
                    fig = create_coverage_heatmap(result)
                else:
                    fig = create_path_figure(
                        result,
                        analysis_mode=analysis_mode,
                        animation_progress=animation_progress,
                        title=f"Run #{st.session_state.run_count}"
                    )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure parameters and click 'Run Simulation' to start")

                preview_params = create_params_from_inputs()
                fig = go.Figure()
                fig.add_shape(
                    type="rect", x0=0, y0=0,
                    x1=preview_params.pool_width, y1=preview_params.pool_height,
                    line=dict(color="#1e3a5f", width=3),
                    fillcolor="rgba(200, 230, 255, 0.3)"
                )
                fig.update_layout(
                    title="Pool Preview",
                    xaxis=dict(title="Width (m)", range=[-1, preview_params.pool_width + 1]),
                    yaxis=dict(title="Length (m)", range=[-1, preview_params.pool_height + 1],
                              scaleanchor="x"),
                    height=500, width=400,
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Right column - Stats and Export
        with right_col:
            if result is not None:
                display_single_run_stats(result)

                st.markdown("---")

                st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)

                export_col1, export_col2 = st.columns(2)

                with export_col1:
                    csv_content = generate_single_run_csv(result)
                    st.download_button(
                        label="Export Single Run CSV",
                        data=csv_content,
                        file_name="simulation_single_run.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with export_col2:
                    if st.session_state.batch_result is not None:
                        batch_csv = generate_batch_csv(st.session_state.batch_result)
                        num_runs_val = st.session_state.batch_result.statistics.num_runs
                        st.download_button(
                            label="Export Batch CSV",
                            data=batch_csv,
                            file_name=f"simulation_batch_{num_runs_val}_runs.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.button("Export Batch CSV", disabled=True, use_container_width=True,
                                 help="Run a batch simulation first")

                st.markdown("---")
                display_event_log(result)
            else:
                st.markdown('<p class="section-header">Statistics</p>', unsafe_allow_html=True)
                st.info("Run a simulation to see statistics")

    # === PARAMETER SCANNER TAB ===
    with tab_scan:
        run_scanner_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        <p>Boat Random Walk Simulator | Master's Project Tool</p>
        <p>Correlated random walk | Moran's I spatial autocorrelation | Coverage analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
