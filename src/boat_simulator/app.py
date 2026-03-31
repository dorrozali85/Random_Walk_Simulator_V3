"""
Boat Simulator - Streamlit Application
A simulation tool for modeling a robotic boat performing correlated random walk
in a rectangular pool, with water sampling and statistical analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from datetime import datetime

# Import simulation modules
from simulation.engine import SimulationParams, run_single_simulation
from simulation.statistics import calculate_all_statistics
from simulation.batch import run_batch_simulation
from visualization.plotting import create_path_figure, create_animated_figure, create_coverage_heatmap
from export.csv_logger import generate_single_run_csv, generate_batch_csv, get_csv_filename

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
        'animation_progress': 1.0,
        'analysis_mode': False,
        'show_animation': False,
        'run_count': 0
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
        edge_buffer=st.session_state.edge_buffer
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


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚤 Boat Random Walk Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Correlated random walk simulation with spatial statistics analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Parameters
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        
        st.subheader("Pool Dimensions")
        st.number_input("Pool Width (m)", min_value=1.0, max_value=100.0, value=12.5, 
                       step=0.5, key="pool_width")
        st.number_input("Pool Height (m)", min_value=1.0, max_value=100.0, value=25.0, 
                       step=0.5, key="pool_height")
        
        st.subheader("Movement Parameters")
        st.number_input("Initial Angle (°)", min_value=0.0, max_value=360.0, value=45.0, 
                       step=5.0, key="alpha")
        st.number_input("Min Delta Angle (°)", min_value=0.0, max_value=90.0, value=25.0, 
                       step=5.0, key="min_delta")
        st.number_input("Max Delta Angle (°)", min_value=0.0, max_value=90.0, value=45.0, 
                       step=5.0, key="max_delta")
        st.number_input("Cruise Speed (m/s)", min_value=0.01, max_value=2.0, value=0.2, 
                       step=0.05, key="cruise_speed")
        
        st.subheader("Edge Behavior")
        st.number_input("Slowdown Factor", min_value=0.1, max_value=1.0, value=0.5, 
                       step=0.1, key="slowdown_factor")
        st.number_input("Edge Buffer (m)", min_value=0.1, max_value=5.0, value=0.5, 
                       step=0.1, key="edge_buffer")
        
        st.subheader("Sampling")
        st.number_input("Sample Interval (min)", min_value=1.0, max_value=60.0, value=10.0, 
                       step=1.0, key="sample_interval")
        st.number_input("Max Samples", min_value=1, max_value=50, value=5, 
                       step=1, key="max_samples")
        
        st.markdown("---")
        
        st.subheader("Batch Settings")
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=100, value=1, 
                                   step=1, key="num_runs")
        
        # Run button
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            params = create_params_from_inputs()
            
            if num_runs == 1:
                # Single run
                with st.spinner("Running simulation..."):
                    result = run_single_simulation(params)
                    result = calculate_all_statistics(result)
                    st.session_state.simulation_result = result
                    st.session_state.batch_result = None
                    st.session_state.run_count += 1
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
                st.session_state.simulation_result = batch_result.runs[-1]  # Last run for visualization
                st.session_state.run_count += 1
                
                progress_bar.empty()
                status_text.empty()
            
            st.success("✅ Simulation complete!")
        
        # Reset button
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.simulation_result = None
            st.session_state.batch_result = None
            st.session_state.animation_progress = 1.0
            st.session_state.analysis_mode = False
            st.rerun()
    
    # Main content area - Three columns
    if st.session_state.batch_result is not None:
        # Show bulk stats panel when batch run exists
        left_col, center_col, right_col = st.columns([1.2, 1.5, 1.3])
        
        with left_col:
            display_bulk_stats(st.session_state.batch_result)
    else:
        center_col, right_col = st.columns([1.5, 1.5])
    
    result = st.session_state.simulation_result
    
    # Center column - Visualization
    with center_col if st.session_state.batch_result else center_col:
        st.markdown('<p class="section-header">🗺️ Visualization</p>', unsafe_allow_html=True)
        
        if result is not None:
            # View controls
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
            
            # Create and display figure
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
            st.info("👆 Configure parameters and click 'Run Simulation' to start")
            
            # Show empty pool preview
            preview_params = create_params_from_inputs()
            import plotly.graph_objects as go
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
            # Single run statistics
            display_single_run_stats(result)
            
            st.markdown("---")
            
            # Export buttons
            st.markdown('<p class="section-header">💾 Export</p>', unsafe_allow_html=True)
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # Single run CSV
                csv_content = generate_single_run_csv(result)
                st.download_button(
                    label="📥 Export Single Run CSV",
                    data=csv_content,
                    file_name="simulation_single_run.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col2:
                # Batch CSV (if available)
                if st.session_state.batch_result is not None:
                    batch_csv = generate_batch_csv(st.session_state.batch_result)
                    num_runs = st.session_state.batch_result.statistics.num_runs
                    st.download_button(
                        label="📥 Export Batch CSV",
                        data=batch_csv,
                        file_name=f"simulation_batch_{num_runs}_runs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.button("📥 Export Batch CSV", disabled=True, use_container_width=True,
                             help="Run a batch simulation first")
            
            st.markdown("---")
            
            # Event log
            display_event_log(result)
        else:
            st.markdown('<p class="section-header">📊 Statistics</p>', unsafe_allow_html=True)
            st.info("Run a simulation to see statistics")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        <p>Boat Random Walk Simulator | Master's Project Tool</p>
        <p>Correlated random walk • Moran's I spatial autocorrelation • Coverage analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
