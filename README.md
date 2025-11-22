# Room Thermostat Simulation

A Python-based 3D thermal simulation of a room equipped with a radiator, a window, and a thermostat. The project demonstrates heat diffusion, convection (simulated via vertical advection), and feedback control using a PID controller.

## Features

*   **3D Heat Simulation**: Uses the Finite Difference Method to solve the Heat Equation with a vertical advection term to simulate hot air rising.
*   **PID Control**: A fully adjustable PID controller regulates the radiator to maintain a desired room temperature.
*   **Interactive Visualization**: Real-time 3D plotting using Matplotlib.
*   **Dynamic Controls**:
    *   **Set Temp**: Adjust the target temperature.
    *   **Win Flow**: Control the rate of cold air entering through the window.
    *   **Win Temp**: Set the outside temperature.
    *   **Speed**: Accelerate the simulation up to 1024x.
    *   **PID Gains**: Tune $K_p, K_i, K_d$ values on the fly.

## Requirements

*   Python 3.x
*   `numpy`
*   `matplotlib`

## Installation

1.  Clone the repository or download the files.
2.  Install dependencies:
    ```bash
    pip install numpy matplotlib
    ```

## Usage

Run the simulation script:

```bash
python simulation.py
```

### Controls
*   **Sliders**: Drag to adjust parameters.
*   **Text Boxes**: Click to edit PID gains (press Enter or click away to apply).
*   **3D Plot**: Click and drag to rotate the view.

## Physics Model

The simulation solves the 3D Heat Equation with an added advection term for buoyancy:

$$ \frac{\partial T}{\partial t} = \alpha \nabla^2 T - v_{z} \frac{\partial T}{\partial z} + Q_{source} - Q_{sink} $$

*   **Radiator**: Modeled as a heat source on the left wall.
*   **Window**: Modeled as a heat sink on the back wall (cooling proportional to flow rate and temperature difference).
*   **Walls**: Adiabatic (insulated) boundary conditions.
