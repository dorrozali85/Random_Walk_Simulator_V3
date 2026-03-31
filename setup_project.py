#!/usr/bin/env python3
"""
Project setup script for Boat Random Walk Simulator
This script prepares the project for use with Claude Code
"""

import os
import shutil
from pathlib import Path

def create_project_structure():
    """Create the recommended project structure for Claude Code"""
    
    # Define the project structure
    directories = [
        "src/boat_simulator",
        "src/boat_simulator/simulation", 
        "src/boat_simulator/visualization",
        "src/boat_simulator/export",
        "src/boat_simulator/utils",
        "tests",
        "docs",
        "examples", 
        "config",
        "scripts",
        "data/raw",
        "data/processed",
        "results",
        "screenshots"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/boat_simulator/__init__.py",
        "src/boat_simulator/simulation/__init__.py",
        "src/boat_simulator/visualization/__init__.py", 
        "src/boat_simulator/export/__init__.py",
        "src/boat_simulator/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created __init__.py: {init_file}")

def copy_existing_files():
    """Copy existing simulator files to the new structure"""
    
    # Define file mappings from current structure to new structure
    file_mappings = {
        "app.py": "src/boat_simulator/app.py",
        "simulation/engine.py": "src/boat_simulator/simulation/engine.py",
        "simulation/statistics.py": "src/boat_simulator/simulation/statistics.py", 
        "simulation/batch.py": "src/boat_simulator/simulation/batch.py",
        "simulation/__init__.py": "src/boat_simulator/simulation/__init__.py",
        "visualization/plotting.py": "src/boat_simulator/visualization/plotting.py",
        "visualization/__init__.py": "src/boat_simulator/visualization/__init__.py",
        "export/csv_logger.py": "src/boat_simulator/export/csv_logger.py", 
        "export/__init__.py": "src/boat_simulator/export/__init__.py",
        "tests/test_simulation.py": "tests/test_simulation.py",
        "tests/test_statistics.py": "tests/test_statistics.py",
        "tests/test_batch.py": "tests/test_batch.py"
    }
    
    # Check if we're in the boat_simulator directory
    if not os.path.exists("simulation"):
        print("⚠️  Run this script from the boat_simulator directory")
        return False
        
    # Copy files
    for src, dst in file_mappings.items():
        if os.path.exists(src):
            # Create destination directory if it doesn't exist
            dst_dir = os.path.dirname(dst)
            Path(dst_dir).mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src, dst)
            print(f"✓ Copied {src} -> {dst}")
        else:
            print(f"⚠️  File not found: {src}")
    
    return True

def create_example_files():
    """Create example files for Claude Code to understand the project"""
    
    # Create basic simulation example
    basic_example = '''#!/usr/bin/env python3
"""
Basic simulation example for Boat Random Walk Simulator
Run with: python examples/basic_simulation.py
"""

import sys
sys.path.insert(0, 'src')

from boat_simulator.simulation.engine import SimulationParams, run_single_simulation
from boat_simulator.simulation.statistics import calculate_all_statistics

def main():
    """Run a basic simulation and print results"""
    
    # Create simulation parameters
    params = SimulationParams(
        pool_width=12.5,
        pool_height=25.0,
        max_samples=8,
        sample_interval=10.0
    )
    
    # Run simulation
    print("Running simulation...")
    result = run_single_simulation(params, seed=42)
    result = calculate_all_statistics(result)
    
    # Print results
    print(f"\\n=== Simulation Results ===")
    print(f"Total time: {result.total_time/60:.1f} minutes")
    print(f"Samples collected: {len(result.samples)}")
    print(f"Coverage: {result.coverage_percent:.1f}%")
    print(f"Moran's I (X): {result.morans_i_x:.3f}")
    print(f"Moran's I (Y): {result.morans_i_y:.3f}")
    print(f"Wall hits: {result.num_wall_hits}")
    
if __name__ == "__main__":
    main()
'''
    
    with open("examples/basic_simulation.py", "w") as f:
        f.write(basic_example)
    
    print("✓ Created examples/basic_simulation.py")

def create_claude_code_instructions():
    """Create specific instructions for Claude Code"""
    
    instructions = """# Claude Code Instructions

## Project Overview
This is a Boat Random Walk Simulator for autonomous robotic sampling research.

## Key Components
- `src/boat_simulator/simulation/` - Core simulation engine
- `src/boat_simulator/visualization/` - Plotting and visualization
- `src/boat_simulator/app.py` - Main Streamlit application
- `tests/` - Unit tests
- `examples/` - Usage examples

## Common Development Tasks

### Run the application
```bash
streamlit run src/boat_simulator/app.py
```

### Run tests
```bash
python -m pytest tests/ -v
```

### Run a basic simulation
```bash
python examples/basic_simulation.py
```

### Create new visualizations
The visualization module can be extended in `src/boat_simulator/visualization/plotting.py`

### Add new statistical analysis
Extend `src/boat_simulator/simulation/statistics.py` for new metrics

## Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write unit tests for new features
- Update examples when adding new functionality

## Areas for Enhancement
1. **Parameter Optimization** - Add automated parameter tuning
2. **3D Simulation** - Extend to 3D environments
3. **Real-time Integration** - Connect to actual boat hardware
4. **Machine Learning** - Add predictive path optimization
5. **Multi-boat Coordination** - Simulate multiple boats

## Academic Context
This simulator is designed for master's/PhD research in:
- Autonomous vehicle path planning
- Water quality sampling strategies
- Spatial statistics and coverage optimization
- Robotic system optimization
"""
    
    with open("CLAUDE_CODE_GUIDE.md", "w") as f:
        f.write(instructions)
    
    print("✓ Created CLAUDE_CODE_GUIDE.md")

def main():
    """Main setup function"""
    print("🚤 Setting up Boat Random Walk Simulator for Claude Code...")
    print()
    
    # Create project structure
    create_project_structure()
    print()
    
    # Copy existing files if available
    if copy_existing_files():
        print()
    
    # Create example files
    create_example_files()
    print()
    
    # Create Claude Code instructions
    create_claude_code_instructions()
    print()
    
    print("✅ Project setup complete!")
    print()
    print("Next steps:")
    print("1. Copy your current boat_simulator files to the new structure")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Test the setup: python examples/basic_simulation.py")
    print("4. Initialize git repository: git init")
    print("5. Ready for Claude Code!")

if __name__ == "__main__":
    main()
