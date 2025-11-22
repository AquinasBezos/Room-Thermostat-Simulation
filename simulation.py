import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=20.0, output_limits=(0, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self._prev_error = 0.0
        self._integral = 0.0
        
    def update(self, measured_value, dt):
        error = self.setpoint - measured_value
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self._integral += error * dt
        # Anti-windup: clamp integral? Simple version for now:
        # Let's just clamp the output, but maybe clamp integral if needed.
        i_term = self.ki * self._integral
        
        # Derivative
        d_term = self.kd * (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        
        output = p_term + i_term + d_term
        
        # Clamp output
        min_out, max_out = self.output_limits
        output = max(min_out, min(max_out, output))
        
        return output

class Room:
    def __init__(self, nx=20, ny=20, nz=20, alpha=0.1, advection_speed=0.05):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.alpha = alpha # Thermal diffusivity
        self.vz = advection_speed # Vertical advection speed (heat rises)
        
        # Initialize temperature grid (ambient 15 degrees)
        self.T = np.ones((nx, ny, nz)) * 15.0
        
        # Locations
        # Radiator: Near Left wall (x=1), bottom half
        self.radiator_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.radiator_mask[1, :, :nz//3] = True
        
        # Window: Near Back wall (y=ny-2), middle height
        self.window_mask = np.zeros((nx, ny, nz), dtype=bool)
        # Window roughly in the middle of the back wall
        y_win = ny - 2
        x_start, x_end = nx//3, 2*nx//3
        z_start, z_end = nz//3, 2*nz//3
        self.window_mask[x_start:x_end, y_win, z_start:z_end] = True
        
        # Thermostat: Near Right wall (x=nx-2), mid height
        self.thermostat_pos = (nx-2, ny//2, nz//2)
        
    def update(self, dt, radiator_power, window_flow, window_temp):
        # 3D Heat Equation with Advection:
        # dT/dt = alpha * Laplacian(T) - vz * dT/dz + Source - Sink
        
        T = self.T
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # Laplacian (Finite Difference)
        # d2T/dx2 + d2T/dy2 + d2T/dz2
        d2x = np.roll(T, 1, axis=0) - 2*T + np.roll(T, -1, axis=0)
        d2y = np.roll(T, 1, axis=1) - 2*T + np.roll(T, -1, axis=1)
        d2z = np.roll(T, 1, axis=2) - 2*T + np.roll(T, -1, axis=2)
        
        # Handle boundaries for Laplacian (Adiabatic/Insulated = 0 gradient)
        # Simple trick: enforce boundaries after update, or just zero out the Laplacian at edges
        # For simplicity in this visual sim, we'll just let 'roll' wrap around but then fix edges?
        # No, 'roll' wraps, which implies periodic boundaries. We want insulated.
        # Better manual calculation or masking.
        # Let's use slicing for speed.
        
        laplacian = np.zeros_like(T)
        laplacian[1:-1, 1:-1, 1:-1] = (
            (T[2:, 1:-1, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) +
            (T[1:-1, 2:, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) +
            (T[1:-1, 1:-1, 2:] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2])
        )
        
        # Advection (Upwind scheme for stability)
        # - vz * dT/dz. If vz > 0 (up), use backward difference in z
        advection = np.zeros_like(T)
        if self.vz > 0:
            advection[:, :, 1:] = -self.vz * (T[:, :, 1:] - T[:, :, :-1])
        
        # Update
        delta_T = dt * (self.alpha * laplacian + advection)
        self.T += delta_T
        
        # Apply Source (Radiator)
        # Power adds heat directly
        # radiator_power is 0-100. Scale it to reasonable temp rise per step
        heat_input = radiator_power * 0.05 * dt 
        self.T[self.radiator_mask] += heat_input
        
        # Apply Sink (Window)
        # Cooling proportional to flow * (T - T_window)
        # window_flow is 0-1.
        mask = self.window_mask
        cooling = window_flow * 0.1 * dt * (self.T[mask] - window_temp)
        self.T[mask] -= cooling
        
        # Enforce Boundaries (Adiabatic - just copy neighbors to edges)
        # This is a crude approximation for insulated walls
        self.T[0, :, :] = self.T[1, :, :]
        self.T[-1, :, :] = self.T[-2, :, :]
        self.T[:, 0, :] = self.T[:, 1, :]
        self.T[:, -1, :] = self.T[:, -2, :] # Note: Window is on this wall, but we applied sink *at* the wall nodes
        self.T[:, :, 0] = self.T[:, :, 1]
        self.T[:, :, -1] = self.T[:, :, -2]
        
    def get_thermostat_reading(self):
        return self.T[self.thermostat_pos]

# --- Visualization & Main Loop ---

def main():
    # Parameters
    NX, NY, NZ = 15, 15, 15
    DT = 0.1
    
    room = Room(nx=NX, ny=NY, nz=NZ, alpha=0.8, advection_speed=0.2)
    pid = PIDController(kp=5.0, ki=0.1, kd=0.5, setpoint=22.0)
    
    # Setup Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Helper to plot a surface
    def plot_wall(ax, X, Y, Z, T_vals):
        # Normalize T for color
        norm = plt.Normalize(10, 30)
        # Facecolors must match X, Y shape. 
        # T_vals is (NY, NX) usually.
        colors = plt.cm.coolwarm(norm(T_vals))
        return ax.plot_surface(X, Y, Z, facecolors=colors, shade=False, alpha=0.8)

    # UI Controls
    ax_temp = plt.axes([0.25, 0.15, 0.65, 0.03])
    s_temp = Slider(ax_temp, 'Set Temp', 10.0, 30.0, valinit=22.0)
    
    ax_flow = plt.axes([0.25, 0.10, 0.65, 0.03])
    s_flow = Slider(ax_flow, 'Win Flow', 0.0, 1.0, valinit=0.5)
    
    ax_wtemp = plt.axes([0.25, 0.05, 0.65, 0.03])
    s_wtemp = Slider(ax_wtemp, 'Win Temp', -10.0, 20.0, valinit=5.0)
    
    ax_speed = plt.axes([0.25, 0.01, 0.65, 0.03])
    s_speed = Slider(ax_speed, 'Speed 2^x', 0, 10, valinit=0, valstep=1)
    
    ax_kp = plt.axes([0.05, 0.6, 0.1, 0.03])
    t_kp = TextBox(ax_kp, 'Kp', initial="5.0")
    
    ax_ki = plt.axes([0.05, 0.55, 0.1, 0.03])
    t_ki = TextBox(ax_ki, 'Ki', initial="0.1")
    
    ax_kd = plt.axes([0.05, 0.5, 0.1, 0.03])
    t_kd = TextBox(ax_kd, 'Kd', initial="0.5")
    
    # Status Text
    status_text = fig.text(0.02, 0.9, "", fontsize=10, verticalalignment='top')

    # Track simulation time
    sim_time = 0.0

    def update(frame):
        nonlocal sim_time
        try:
            # 1. Update Physics
            try:
                kp = float(t_kp.text)
                ki = float(t_ki.text)
                kd = float(t_kd.text)
            except ValueError:
                kp, ki, kd = 5.0, 0.1, 0.5 
                
            pid.kp, pid.ki, pid.kd = kp, ki, kd
            pid.setpoint = s_temp.val
            
            # Speed multiplier
            speed_steps = int(2 ** s_speed.val)
            
            current_temp = 0
            power = 0
            
            for _ in range(speed_steps):
                current_temp = room.get_thermostat_reading()
                power = pid.update(current_temp, DT)
                room.update(DT, power, s_flow.val, s_wtemp.val)
                sim_time += DT
            
            # 2. Update Visualization
            ax.clear()
            ax.set_xlim(0, NX)
            ax.set_ylim(0, NY)
            ax.set_zlim(0, NZ)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z (Height)')
            
            # Plot Walls
            # Left Wall (x=0) - Radiator
            # Meshgrid: Y varies, Z varies. X is 0.
            y_range = np.arange(NY)
            z_range = np.arange(NZ)
            Y, Z = np.meshgrid(y_range, z_range)
            X = np.zeros_like(Y)
            # T at x=0 is room.T[0, :, :] which is (NY, NZ).
            # Meshgrid order: Y is axis 0 of result? No.
            # meshgrid(y, z) -> Y has shape (len(z), len(y)) = (NZ, NY).
            # room.T[0, :, :] has shape (NY, NZ).
            # So we need transpose.
            T_vals = room.T[0, :, :].T 
            plot_wall(ax, X, Y, Z, T_vals)
            
            # Back Wall (y=NY-1) - Window
            x_range = np.arange(NX)
            X, Z = np.meshgrid(x_range, z_range) # Shape (NZ, NX)
            Y = np.full_like(X, NY-1)
            T_vals = room.T[:, -1, :].T # (NX, NZ) -> (NZ, NX)
            plot_wall(ax, X, Y, Z, T_vals)
            
            # Floor (z=0)
            X, Y = np.meshgrid(x_range, y_range) # Shape (NY, NX)
            Z = np.zeros_like(X)
            T_vals = room.T[:, :, 0].T # (NX, NY) -> (NY, NX)
            plot_wall(ax, X, Y, Z, T_vals)

            # Markers
            # Radiator (Red point on Left Wall)
            ax.scatter([0], [NY/2], [NZ/4], c='red', s=200, label='Radiator', depthshade=False)
            # Window (Blue point on Back Wall)
            ax.scatter([NX/2], [NY-1], [NZ/2], c='blue', s=200, label='Window', depthshade=False)
            # Thermostat (Green point on Right Wall)
            ax.scatter([NX-1], [NY/2], [NZ/2], c='green', s=200, label='Thermostat', depthshade=False)
            
            # ax.legend() # Legend can be slow/buggy in 3D animation sometimes, try without if needed
            
            status_text.set_text(
                f"Time: {sim_time:.1f}s\n"
                f"Thermostat: {current_temp:.2f}C\n"
                f"Setpoint: {pid.setpoint:.1f}C\n"
                f"Radiator Power: {power:.1f}%\n"
                f"Window Flow: {s_flow.val:.2f}"
            )
        except Exception as e:
            print(f"Error in update: {e}")
            import traceback
            traceback.print_exc()

    # Call update once to initialize plot
    update(0)
    
    ani = FuncAnimation(fig, update, frames=range(10000), interval=100, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
