# Version: v3.2  |  Date: 2026-04-03
"""
Boat Simulation Engine
Implements correlated random walk with physical bumper wall behaviour and time-based sampling.
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
    
    # Wall behaviour parameters
    boat_width: float = 0.6           # physical boat width (m); bumper radius = boat_width/2
    stop_time: float = 2.0            # seconds boat stops at wall after impact
    acceleration: float = 0.1        # m/s² speed ramp-up after stop

    # Fixed parameters (not tunable per spec)
    dt: float = 0.1                   # time step in seconds
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
        bumper_r = p.boat_width / 2   # min distance from boat centre to any wall

        # Initialize boat position and heading
        meter_x = max(bumper_r, 0.5)
        meter_y = bumper_r            # start away from bottom wall by bumper radius
        angle_rad = math.radians(p.alpha)
        hx = math.cos(angle_rad)       # heading unit vector x
        hy = math.sin(angle_rad)       # heading unit vector y
        actual_speed = p.cruise_speed  # current speed (ramps from 0 after wall hit)
        vx = hx * actual_speed
        vy = hy * actual_speed
        
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
            
            target_speed = p.cruise_speed * (p.slowdown_factor if near_edge else 1.0)

            # Ramp actual_speed toward target (acceleration from 0 after wall stop)
            if actual_speed < target_speed:
                actual_speed = min(actual_speed + p.acceleration * p.dt, target_speed)
            elif actual_speed > target_speed:
                actual_speed = target_speed   # snap down when entering edge buffer

            # Build velocity from heading direction and current speed
            vx = hx * actual_speed
            vy = hy * actual_speed
            
            # Update position
            new_x = meter_x + vx * p.dt
            new_y = meter_y + vy * p.dt
            
            # Wall collision — physical bumper model: stop, rotate heading, accelerate away
            wall_hit = False
            angle_change = 0.0
            wnx, wny = 0.0, 0.0   # wall outward normal (first hit wins for rotation sign)

            # Left wall (trigger when bumper face reaches wall)
            if new_x < bumper_r:
                new_x = bumper_r
                wall_hit = True
                wnx, wny = -1.0, 0.0
            elif new_x > p.pool_width - bumper_r:
                new_x = p.pool_width - bumper_r
                wall_hit = True
                wnx, wny = 1.0, 0.0

            # Bottom/top walls (x-wall takes priority for rotation if corner hit)
            if new_y < bumper_r:
                new_y = bumper_r
                if not wall_hit:
                    wall_hit = True
                    wnx, wny = 0.0, -1.0
            elif new_y > p.pool_height - bumper_r:
                new_y = p.pool_height - bumper_r
                if not wall_hit:
                    wall_hit = True
                    wnx, wny = 0.0, 1.0

            if wall_hit:
                # Bumper side: cross product of approach heading and wall outward normal
                # cross > 0 → wall to LEFT  → LEFT microswitch  → CW  (sign = -1)
                # cross < 0 → wall to RIGHT → RIGHT microswitch → CCW (sign = +1)
                # cross = 0 → dead center                        → CW  (sign = -1)
                cross = hx * wny - hy * wnx
                sign = -1 if cross >= 0 else 1

                delta_deg = self.rng.uniform(p.min_delta, p.max_delta)
                angle_change = delta_deg * sign

                # Rotate heading unit vector
                ar = math.radians(angle_change)
                cos_a, sin_a = math.cos(ar), math.sin(ar)
                new_hx = hx * cos_a - hy * sin_a
                new_hy = hx * sin_a + hy * cos_a
                # Re-normalise to guard against floating-point drift
                h_mag = math.sqrt(new_hx**2 + new_hy**2)
                if h_mag > 0:
                    hx, hy = new_hx / h_mag, new_hy / h_mag

                # Escape guarantee: if heading still points into the wall after rotation
                # (rotation angle was too small), reflect the wall-normal component so
                # the boat always leaves the wall after one event.
                # Physical interpretation: the robot keeps turning until it faces away.
                # dot_out > 0 means heading has a component along the outward normal (into wall)
                dot_out = hx * wnx + hy * wny
                if dot_out >= 0:
                    hx -= 2 * dot_out * wnx   # reflect wall-normal component
                    hy -= 2 * dot_out * wny   # (reflection preserves unit length)

                # Boat stops at wall for stop_time seconds, then accelerates away
                actual_speed = 0.0
                sim_time += p.stop_time
                sample_accumulator += p.stop_time

                result.num_wall_hits += 1
                result.events.append(SimulationEvent(
                    timestamp=sim_time,
                    event_type='WallHit',
                    position_x=new_x,
                    position_y=new_y,
                    angle_change=angle_change
                ))
            
            # Update position (centre always stays bumper_r from every wall)
            meter_x = np.clip(new_x, bumper_r, p.pool_width  - bumper_r)
            meter_y = np.clip(new_y, bumper_r, p.pool_height - bumper_r)
            
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
