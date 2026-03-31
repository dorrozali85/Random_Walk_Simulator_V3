"""
Boat Simulation Engine
Implements correlated random walk with wall bouncing and time-based sampling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


@dataclass
class SimulationParams:
    """All tunable simulation parameters with defaults from spec."""
    pool_width: float = 12.5          # meters
    pool_height: float = 25.0         # meters
    alpha: float = 45.0               # initial direction angle (degrees)
    min_delta: float = 25.0           # min random angle change on hit (degrees)
    max_delta: float = 45.0           # max random angle change on hit (degrees)
    sample_interval: float = 10.0     # minutes between samples
    max_samples: int = 5              # max number of samples (stops sim)
    cruise_speed: float = 0.2         # m/s normal speed
    slowdown_factor: float = 0.5      # speed multiplier near edge
    edge_buffer: float = 0.5          # meters - distance to start slowdown
    
    # Fixed parameters (not tunable per spec)
    dt: float = 0.1                   # time step in seconds
    turn_delay: float = 1.5           # seconds after hit
    max_steps: int = 100000           # safety limit
    
    # Derived
    @property
    def sample_interval_seconds(self) -> float:
        return self.sample_interval * 60.0


@dataclass
class SimulationEvent:
    """Single logged event during simulation."""
    timestamp: float          # seconds
    event_type: str           # 'Start', 'WallHit', 'WaterSample'
    position_x: float         # meters
    position_y: float         # meters
    angle_change: float = 0.0 # degrees (for wall hits)


@dataclass
class PathPoint:
    """Single point in the boat's path for visualization."""
    time: float
    x: float
    y: float


@dataclass
class SamplePoint:
    """Water sample location (blue dot)."""
    time: float
    x: float
    y: float
    sample_number: int


@dataclass
class SimulationResult:
    """Complete result of a single simulation run."""
    params: SimulationParams
    path: List[PathPoint] = field(default_factory=list)
    samples: List[SamplePoint] = field(default_factory=list)
    events: List[SimulationEvent] = field(default_factory=list)
    
    # Statistics (computed after simulation)
    morans_i_x: float = 0.0
    morans_i_y: float = 0.0
    coverage_percent: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    avg_distance: float = 0.0
    lag1_correlation: float = 0.0
    
    # Metadata
    total_time: float = 0.0
    num_wall_hits: int = 0


class BoatSimulator:
    """
    Core simulation engine for the boat's correlated random walk.
    """
    
    def __init__(self, params: SimulationParams, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        
    def run(self) -> SimulationResult:
        """Execute a complete simulation and return results."""
        p = self.params
        result = SimulationResult(params=p)
        
        # Initialize boat position and velocity
        meter_x = 0.5
        meter_y = 0.0
        angle_rad = math.radians(p.alpha)
        speed = p.cruise_speed
        vx = speed * math.cos(angle_rad)
        vy = speed * math.sin(angle_rad)
        
        # Time tracking
        sim_time = 0.0
        sample_accumulator = 0.0
        sample_count = 0
        
        # Log start event
        result.events.append(SimulationEvent(
            timestamp=0.0,
            event_type='Start',
            position_x=meter_x,
            position_y=meter_y,
            angle_change=p.alpha
        ))
        
        # Store initial path point
        result.path.append(PathPoint(time=0.0, x=meter_x, y=meter_y))
        
        # Main simulation loop
        for step in range(p.max_steps):
            if sample_count >= p.max_samples:
                break
                
            # Check if near edge - apply slowdown
            near_edge = (
                meter_x < p.edge_buffer or 
                meter_x > p.pool_width - p.edge_buffer or
                meter_y < p.edge_buffer or 
                meter_y > p.pool_height - p.edge_buffer
            )
            
            current_speed = p.cruise_speed * (p.slowdown_factor if near_edge else 1.0)
            
            # Normalize velocity to current speed
            vel_mag = math.sqrt(vx**2 + vy**2)
            if vel_mag > 0:
                vx = (vx / vel_mag) * current_speed
                vy = (vy / vel_mag) * current_speed
            
            # Update position
            new_x = meter_x + vx * p.dt
            new_y = meter_y + vy * p.dt
            
            # Wall collision detection and handling
            wall_hit = False
            angle_change = 0.0
            
            # Left wall
            if new_x < 0:
                new_x = -new_x
                vx = -vx
                wall_hit = True
                
            # Right wall
            if new_x > p.pool_width:
                new_x = 2 * p.pool_width - new_x
                vx = -vx
                wall_hit = True
                
            # Bottom wall
            if new_y < 0:
                new_y = -new_y
                vy = -vy
                wall_hit = True
                
            # Top wall
            if new_y > p.pool_height:
                new_y = 2 * p.pool_height - new_y
                vy = -vy
                wall_hit = True
            
            # Apply random angle change on wall hit
            if wall_hit:
                # Random delta between min and max, with random sign
                delta_deg = self.rng.uniform(p.min_delta, p.max_delta)
                sign = self.rng.choice([-1, 1])
                angle_change = delta_deg * sign
                
                # Apply rotation to velocity
                angle_rad = math.radians(angle_change)
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                new_vx = vx * cos_a - vy * sin_a
                new_vy = vx * sin_a + vy * cos_a
                vx, vy = new_vx, new_vy
                
                # Add turn delay
                sim_time += p.turn_delay
                sample_accumulator += p.turn_delay
                
                result.num_wall_hits += 1
                result.events.append(SimulationEvent(
                    timestamp=sim_time,
                    event_type='WallHit',
                    position_x=new_x,
                    position_y=new_y,
                    angle_change=angle_change
                ))
            
            # Update position
            meter_x = np.clip(new_x, 0, p.pool_width)
            meter_y = np.clip(new_y, 0, p.pool_height)
            
            # Advance time
            sim_time += p.dt
            sample_accumulator += p.dt
            
            # Store path point (downsample for memory efficiency)
            if step % 5 == 0:
                result.path.append(PathPoint(time=sim_time, x=meter_x, y=meter_y))
            
            # Check for sampling
            if sample_accumulator >= p.sample_interval_seconds:
                sample_count += 1
                sample_accumulator = 0.0
                
                result.samples.append(SamplePoint(
                    time=sim_time,
                    x=meter_x,
                    y=meter_y,
                    sample_number=sample_count
                ))
                
                result.events.append(SimulationEvent(
                    timestamp=sim_time,
                    event_type='WaterSample',
                    position_x=meter_x,
                    position_y=meter_y,
                    angle_change=0.0
                ))
        
        result.total_time = sim_time
        
        # Ensure final path point is recorded
        if not result.path or result.path[-1].time != sim_time:
            result.path.append(PathPoint(time=sim_time, x=meter_x, y=meter_y))
        
        return result


def run_single_simulation(params: SimulationParams, seed: Optional[int] = None) -> SimulationResult:
    """Convenience function to run a single simulation."""
    simulator = BoatSimulator(params, seed)
    return simulator.run()
