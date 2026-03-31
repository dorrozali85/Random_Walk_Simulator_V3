"""
Visualization Module
Creates Plotly figures for path visualization and analysis mode.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional, Tuple
from simulation.engine import SimulationResult, PathPoint, SamplePoint
from simulation.statistics import find_nearest_neighbors


def create_path_figure(
    result: SimulationResult,
    show_path: bool = True,
    show_samples: bool = True,
    analysis_mode: bool = False,
    animation_progress: float = 1.0,
    title: str = "Boat Simulation"
) -> go.Figure:
    """
    Create a Plotly figure showing the boat's path and samples.
    
    Args:
        result: Simulation result
        show_path: Whether to show the path line
        show_samples: Whether to show sample points
        analysis_mode: If True, show only samples with nearest-neighbor lines
        animation_progress: 0.0 to 1.0, fraction of path/samples to show
        title: Figure title
        
    Returns:
        Plotly Figure object
    """
    params = result.params
    
    # Create figure
    fig = go.Figure()
    
    # Pool boundaries
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=params.pool_width, y1=params.pool_height,
        line=dict(color="#1e3a5f", width=3),
        fillcolor="rgba(200, 230, 255, 0.3)"
    )
    
    # Calculate how much to show based on animation progress
    path_end_idx = int(len(result.path) * animation_progress)
    path_end_idx = max(1, path_end_idx)
    visible_path = result.path[:path_end_idx]
    
    # Determine visible samples based on time
    if visible_path:
        max_time = visible_path[-1].time
        visible_samples = [s for s in result.samples if s.time <= max_time]
    else:
        visible_samples = []
    
    if analysis_mode:
        # Analysis mode: only blue points with pink nearest-neighbor lines
        if len(visible_samples) >= 2:
            neighbors = find_nearest_neighbors(visible_samples)
            
            # Draw nearest-neighbor lines (pink)
            for from_idx, to_idx, dist in neighbors:
                s1, s2 = visible_samples[from_idx], visible_samples[to_idx]
                fig.add_trace(go.Scatter(
                    x=[s1.x, s2.x],
                    y=[s1.y, s2.y],
                    mode='lines',
                    line=dict(color='#ff69b4', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Draw sample points (blue)
            fig.add_trace(go.Scatter(
                x=[s.x for s in visible_samples],
                y=[s.y for s in visible_samples],
                mode='markers+text',
                marker=dict(color='#1e90ff', size=14, symbol='circle',
                           line=dict(color='white', width=2)),
                text=[str(s.sample_number) for s in visible_samples],
                textposition='top center',
                textfont=dict(size=10, color='#1e3a5f'),
                name='Samples',
                hovertemplate='Sample %{text}<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>'
            ))
    else:
        # Normal mode: show path and samples
        if show_path and visible_path:
            fig.add_trace(go.Scatter(
                x=[p.x for p in visible_path],
                y=[p.y for p in visible_path],
                mode='lines',
                line=dict(color='#2c3e50', width=2),
                name='Path',
                hoverinfo='skip'
            ))
            
            # Current position (red dot)
            if visible_path:
                current = visible_path[-1]
                fig.add_trace(go.Scatter(
                    x=[current.x],
                    y=[current.y],
                    mode='markers',
                    marker=dict(color='#e74c3c', size=12, symbol='circle',
                               line=dict(color='white', width=2)),
                    name='Current Position',
                    hovertemplate='Current<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>'
                ))
        
        if show_samples and visible_samples:
            fig.add_trace(go.Scatter(
                x=[s.x for s in visible_samples],
                y=[s.y for s in visible_samples],
                mode='markers+text',
                marker=dict(color='#3498db', size=12, symbol='circle',
                           line=dict(color='white', width=2)),
                text=[str(s.sample_number) for s in visible_samples],
                textposition='top center',
                textfont=dict(size=10, color='#1e3a5f'),
                name='Samples',
                hovertemplate='Sample %{text}<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>'
            ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title="Width (m)",
            range=[-0.5, params.pool_width + 0.5],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="Length (m)",
            range=[-0.5, params.pool_height + 0.5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
        width=450
    )
    
    return fig


def create_animated_figure(result: SimulationResult, num_frames: int = 100) -> go.Figure:
    """
    Create an animated Plotly figure with playback frames.
    
    Args:
        result: Simulation result
        num_frames: Number of animation frames
        
    Returns:
        Plotly Figure with animation frames
    """
    params = result.params
    
    # Create base figure
    fig = go.Figure()
    
    # Pool boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=params.pool_width, y1=params.pool_height,
        line=dict(color="#1e3a5f", width=3),
        fillcolor="rgba(200, 230, 255, 0.3)"
    )
    
    # Initial traces (will be updated in frames)
    # Path line
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='#2c3e50', width=2),
        name='Path'
    ))
    
    # Current position
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='markers',
        marker=dict(color='#e74c3c', size=12, symbol='circle',
                   line=dict(color='white', width=2)),
        name='Current'
    ))
    
    # Samples
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='markers+text',
        marker=dict(color='#3498db', size=12, symbol='circle',
                   line=dict(color='white', width=2)),
        text=[],
        textposition='top center',
        name='Samples'
    ))
    
    # Create frames
    frames = []
    for i in range(num_frames + 1):
        progress = i / num_frames
        path_end_idx = max(1, int(len(result.path) * progress))
        visible_path = result.path[:path_end_idx]
        
        max_time = visible_path[-1].time if visible_path else 0
        visible_samples = [s for s in result.samples if s.time <= max_time]
        
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=[p.x for p in visible_path],
                    y=[p.y for p in visible_path]
                ),
                go.Scatter(
                    x=[visible_path[-1].x] if visible_path else [],
                    y=[visible_path[-1].y] if visible_path else []
                ),
                go.Scatter(
                    x=[s.x for s in visible_samples],
                    y=[s.y for s in visible_samples],
                    text=[str(s.sample_number) for s in visible_samples]
                )
            ],
            name=str(i)
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="left",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {
                             "frame": {"duration": 50, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 0}
                         }]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {
                             "frame": {"duration": 0, "redraw": False},
                             "mode": "immediate",
                             "transition": {"duration": 0}
                         }])
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "steps": [{"args": [[str(k)], {"frame": {"duration": 50, "redraw": True},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}],
                       "label": str(k),
                       "method": "animate"} for k in range(0, num_frames + 1, 5)],
            "x": 0.1,
            "len": 0.8,
            "xanchor": "left",
            "y": -0.1,
            "yanchor": "top",
            "currentvalue": {
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "center"
            },
            "transition": {"duration": 0}
        }]
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text="Boat Simulation - Animated", x=0.5, font=dict(size=16)),
        xaxis=dict(
            title="Width (m)",
            range=[-0.5, params.pool_width + 0.5],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="Length (m)",
            range=[-0.5, params.pool_height + 0.5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=100),
        height=650,
        width=450
    )
    
    return fig


def create_coverage_heatmap(result: SimulationResult, grid_cols: int = 20, grid_rows: int = 10) -> go.Figure:
    """
    Create a heatmap showing coverage intensity.
    """
    params = result.params
    
    # Create coverage grid
    cell_width = params.pool_width / grid_cols
    cell_height = params.pool_height / grid_rows
    visit_count = np.zeros((grid_rows, grid_cols))
    
    # Count visits per cell
    for point in result.path:
        col = int(min(point.x / cell_width, grid_cols - 1))
        row = int(min(point.y / cell_height, grid_rows - 1))
        col = max(0, col)
        row = max(0, row)
        visit_count[row, col] += 1
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=visit_count,
        x=np.linspace(cell_width/2, params.pool_width - cell_width/2, grid_cols),
        y=np.linspace(cell_height/2, params.pool_height - cell_height/2, grid_rows),
        colorscale='Blues',
        colorbar=dict(title='Visits')
    ))
    
    # Add sample points overlay
    if result.samples:
        fig.add_trace(go.Scatter(
            x=[s.x for s in result.samples],
            y=[s.y for s in result.samples],
            mode='markers+text',
            marker=dict(color='red', size=10, symbol='circle',
                       line=dict(color='white', width=2)),
            text=[str(s.sample_number) for s in result.samples],
            textposition='top center',
            textfont=dict(size=10, color='red'),
            name='Samples'
        ))
    
    fig.update_layout(
        title=dict(text="Coverage Heatmap", x=0.5),
        xaxis=dict(title="Width (m)"),
        yaxis=dict(title="Length (m)", scaleanchor="x"),
        height=500,
        width=400
    )
    
    return fig
