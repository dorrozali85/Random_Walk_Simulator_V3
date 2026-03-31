# 🚤 Boat Random Walk Simulator

A comprehensive simulation platform for autonomous robotic boat sampling in confined aquatic environments using Correlated Random Walk (CRW) algorithms.

## 🎯 Project Overview

This project implements a sophisticated simulation of a robotic boat performing water sampling using correlated random walk methodology. The system provides:

- **Physics-based simulation** with realistic boat dynamics
- **Advanced statistical analysis** including Moran's I spatial autocorrelation
- **Interactive visualization** with real-time path tracking
- **Comprehensive reporting** with CSV export capabilities
- **Batch processing** for statistical robustness

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd boat_simulator_project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run src/app.py
```

### Basic Usage
```python
from boat_simulator.simulation import run_single_simulation, SimulationParams

# Create parameters
params = SimulationParams(
    pool_width=12.5,
    pool_height=25.0,
    max_samples=12
)

# Run simulation
result = run_single_simulation(params, seed=42)
print(f"Coverage: {result.coverage_percent:.1f}%")
```

## 📊 Key Features

### Simulation Engine
- **Correlated Random Walk** with wall bouncing
- **Time-based sampling** at fixed intervals
- **Realistic physics** including edge slowdown and turn delays
- **Deterministic results** with seed control

### Statistical Analysis
- **Moran's I** spatial autocorrelation (X and Y axes)
- **Coverage analysis** using grid-based method
- **Distance metrics** between consecutive samples
- **K-Nearest Neighbor** analysis

### Visualization
- **Real-time path plotting** with sample points
- **Interactive controls** with animation slider
- **Analysis mode** showing nearest-neighbor connections
- **Coverage heatmaps** with visit intensity

### Export Capabilities
- **Single run CSV** with complete event log
- **Batch run CSV** with aggregated statistics
- **Publication-ready figures** in high resolution

## 🔧 Configuration

### Default Parameters
```yaml
# config/default_params.yaml
simulation:
  pool_width: 12.5      # meters
  pool_height: 25.0     # meters
  cruise_speed: 0.2     # m/s
  sample_interval: 10.0 # minutes
  max_samples: 12       # count
  
physics:
  min_delta: 25.0       # degrees
  max_delta: 45.0       # degrees
  slowdown_factor: 0.5  # ratio
  edge_buffer: 0.5      # meters
```

## 📈 Use Cases

1. **Academic Research** - Master's thesis, PhD research
2. **Water Quality Monitoring** - Autonomous sampling strategies
3. **Robotics Development** - Path planning algorithms
4. **Environmental Engineering** - Pollution detection systems

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_simulation.py -v
python -m pytest tests/test_statistics.py -v

# Generate coverage report
python -m pytest --cov=boat_simulator tests/
```

## 📝 Examples

### Generate Academic Report
```python
python examples/generate_report.py --samples 12 --runtime 120 --output report.docx
```

### Batch Analysis
```python
python examples/run_batch_analysis.py --runs 50 --export-csv
```

### Parameter Optimization
```python
python examples/optimize_parameters.py --target-coverage 90 --max-time 80
```

## 🎨 Screenshots & Visualization

The `screenshots/` directory contains example outputs:
- `simulation_overview.png` - Full simulation with path and samples
- `coverage_heatmap.png` - Spatial coverage analysis
- `statistical_analysis.png` - Moran's I comparison charts

## 📚 Documentation

- **User Guide**: `docs/user_guide.md`
- **API Reference**: `docs/api_reference.md`
- **Academic Paper**: See comprehensive analysis document

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📋 TODO / Roadmap

- [ ] Add 3D simulation capabilities
- [ ] Implement machine learning optimization
- [ ] Add real-time sensor integration
- [ ] Develop multi-boat coordination
- [ ] Create web-based dashboard

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 References

1. Correlated Random Walk Theory
2. Moran's I Spatial Autocorrelation
3. Autonomous Vehicle Path Planning
4. Water Quality Sampling Methodologies

## 📧 Contact

For questions about this project, please open an issue or contact the development team.

---

**Note for Claude Code**: This project is ready for agentic development. Key areas for enhancement include parameter optimization, advanced statistical analysis, and real-world integration capabilities.
