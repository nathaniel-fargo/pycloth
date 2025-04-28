# Real-Time Cloth Simulation

A Python-based real-time cloth simulation that models a flexible fabric with interactive wind controls and energy visualization.

## Features

- Real-time cloth simulation with physics-based modeling
- Interactive wind controls (x and y directions)
- Energy visualization (kinetic, gravitational potential, and elastic energies)
- Adjustable simulation parameters
- 3D visualization with matplotlib

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Usage

Run the simulation with default parameters:

```bash
python cloth_sim_rt.py
```

### Command Line Arguments

- `--nx`: Number of points in x-direction (default: 20)
- `--nz`: Number of points in z-direction (default: 20)
- `--rest_len`: Rest length of springs (default: 0.05)
- `--mass`: Mass of each point (default: 0.1)
- `--k`: Spring constant (default: 1000)
- `--damping`: Damping coefficient (default: 0.01)
- `--drag_coeff`: Air drag coefficient (default: 0.3)
- `--dt`: Time step (default: 0.02)
- `--fps`: Frames per second (default: 45)
- `--window`: Time window for energy plot (default: 10.0)

Example with custom parameters:

```bash
python cloth_sim_rt.py --nx 30 --nz 30 --mass 0.2 --k 2000
```

## Controls

- Use the sliders at the bottom of the window to control wind direction and magnitude
- The left panel shows the 3D cloth simulation
- The right panel displays the energy components over time:
  - KE: Kinetic Energy
  - GPE: Gravitational Potential Energy
  - Elastic: Elastic Potential Energy
  - Total: Total System Energy

## Physics Model

The simulation includes:
- Spring-mass system for cloth modeling
- Gravity
- Air drag with directional effects
- Linear damping
- Fixed anchor points at the top

## Author

Nathaniel Fargo 