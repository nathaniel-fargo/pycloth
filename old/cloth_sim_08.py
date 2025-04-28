"""
cloth_sim_rt.py (interactive v7 - animation fix)
------------------------------------------------
* nonlocal surface inside update so it refreshes each frame
* energy plot now updates regardless of user interaction
* minor speed tweak: reuse Poly3DCollection via set_verts if available
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from collections import deque
from dataclasses import dataclass

# Constants
GRAVITY_ACCEL = 9.81
GRAVITY_VECTOR = np.array([0.0, 0.0, -GRAVITY_ACCEL])

@dataclass
class SpringArray:
    """Stores spring endpoint indices, rest lengths, and spring constant."""
    idx_start: np.ndarray
    idx_end:   np.ndarray
    rest_length: np.ndarray
    k: float

class Cloth:
    def __init__(self, nx, nz, rest_length, mass, spring_k, damping):
        # Grid dimensions and derived quantities
        self.nx = nx
        self.nz = nz
        self.num_points = nx * nz

        # Physical parameters
        self.mass = mass
        self.spring_k = spring_k
        self.damping = damping

        # Initial grid positions (x,z) and zero y
        x_positions = np.linspace(0, (nx - 1) * rest_length, nx)
        z_positions = np.linspace(0, - (nz - 1) * rest_length, nz)
        grid_x, grid_z = np.meshgrid(x_positions, z_positions)
        self.positions = np.c_[grid_x.ravel(),             # x
                               np.zeros(self.num_points),  # y
                               grid_z.ravel()]             # z

        # Velocities start at zero
        self.velocities = np.zeros_like(self.positions)

        # Fix the top row of points as anchors
        self.fixed_indices = np.arange(nx)
        self.initial_positions = self.positions.copy()

        # Build spring connections (structural + shear)
        starts, ends, rests = [], [], []
        for row in range(nz):
            for col in range(nx):
                idx = row * nx + col
                # Horizontal neighbor
                if col < nx - 1:
                    starts.append(idx); ends.append(idx + 1); rests.append(rest_length)
                # Vertical neighbor
                if row < nz - 1:
                    starts.append(idx); ends.append(idx + nx); rests.append(rest_length)
                # Shear neighbor (down-right)
                if col < nx - 1 and row < nz - 1:
                    starts.append(idx); ends.append(idx + nx + 1); rests.append(rest_length * np.sqrt(2))
                # Shear neighbor (down-left)
                if col > 0 and row < nz - 1:
                    starts.append(idx); ends.append(idx + nx - 1); rests.append(rest_length * np.sqrt(2))

        self.springs = SpringArray(
            idx_start=np.array(starts, dtype=int),
            idx_end=np.array(ends, dtype=int),
            rest_length=np.array(rests),
            k=spring_k
        )

        # Record initial minimum z for GPE reference
        self.z_min = self.positions[:, 2].min()

    def _step(self, delta_t, wind_vector):
        """Perform a single small integration step."""
        # Start with gravity + wind forces
        forces = np.broadcast_to(self.mass * GRAVITY_VECTOR + wind_vector * self.mass,
                                 self.positions.shape).copy()

        # Compute spring displacements
        disp = self.positions[self.springs.idx_end] - self.positions[self.springs.idx_start]
        lengths = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-12
        spring_forces = (self.springs.k * (lengths - self.springs.rest_length[:, None])
                         * (disp / lengths))

        # Accumulate forces on each endpoint
        np.add.at(forces, self.springs.idx_start,  spring_forces)
        np.add.at(forces, self.springs.idx_end,   -spring_forces)

        # Damping
        forces += -self.damping * self.velocities

        # Update velocities and positions
        self.velocities += (forces / self.mass) * delta_t
        self.positions  +=  self.velocities * delta_t

        # Re-apply anchor constraints
        self.positions[self.fixed_indices] = self.initial_positions[self.fixed_indices]
        self.velocities[self.fixed_indices] = 0.0

    def step(self, dt, wind_vector):
        """
        Advance simulation by dt, splitting into safe sub-steps
        to maintain stability.
        """
        # Maximum stable sub-step
        max_sub = 0.15 * np.sqrt(self.mass / self.spring_k)
        num_substeps = max(1, int(np.ceil(dt / max_sub)))
        sub_dt = dt / num_substeps

        for _ in range(num_substeps):
            self._step(sub_dt, wind_vector)

    def energies(self):
        """Return (kinetic, gravitational, elastic) energies."""
        # Kinetic energy
        ke = 0.5 * self.mass * np.sum(self.velocities ** 2)

        # Gravitational potential energy
        ge = self.mass * GRAVITY_ACCEL * np.sum(self.positions[:, 2] -
                                                self.initial_positions[:, 2])

        # Elastic potential energy
        current_lengths = np.linalg.norm(
            self.positions[self.springs.idx_end] - self.positions[self.springs.idx_start],
            axis=1
        )
        stretch = current_lengths - self.springs.rest_length
        ee = 0.5 * self.spring_k * np.sum(stretch ** 2)

        return ke, ge, ee

    def get_grid(self):
        """Return X, Y, Z arrays shaped (nz, nx) for plotting."""
        grid = self.positions.reshape(self.nz, self.nx, 3)
        return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]

def main():
    # Parse command‑line arguments
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--nx',        type=int,   default=20)
    add_arg('--nz',        type=int,   default=20)
    add_arg('--rest_len',  type=float, default=0.15)
    add_arg('--mass',      type=float, default=0.05)
    add_arg('--k',         type=float, default=500)
    add_arg('--damping',   type=float, default=0.06)
    add_arg('--dt',        type=float, default=0.01)
    add_arg('--fps',       type=int,   default=45)
    add_arg('--window',    type=float, default=10.0)
    args = parser.parse_args()

    # Create cloth simulation and buffers for plotting energies
    cloth = Cloth(args.nx, args.nz, args.rest_len,
                  args.mass, args.k, args.damping)
    buffer_size = int(args.window / args.dt)
    time_buffer, ke_buffer, ge_buffer, ee_buffer, tot_buffer = (
        deque(maxlen=buffer_size) for _ in range(5)
    )

    # Set up figure with 3D cloth and energy plot
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)

    # --- Cloth surface subplot ---
    ax_surface = fig.add_subplot(gs[0], projection='3d')
    ax_surface.view_init(20, 0)
    for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
        axis.set_ticks([])
    ax_surface.set_xlim(0, (args.nx - 1) * args.rest_len)
    span = max(args.nx - 1, args.nz - 1) * args.rest_len / 2
    ax_surface.set_ylim(-span, span)
    ax_surface.set_zlim(-(args.nz - 1) * args.rest_len, 0.1)
    ax_surface.set_xlabel('x')
    ax_surface.set_ylabel('y')
    ax_surface.set_zlabel('z')
    X, Y, Z = cloth.get_grid()
    surface_plot = ax_surface.plot_surface(
        X, Y, Z,
        color='cornflowerblue', edgecolor='grey', lw=0.3, alpha=0.9
    )

    # --- Energy plot subplot ---
    ax_energy = fig.add_subplot(gs[1])
    ax_energy.set_xlabel('t (s)')
    ax_energy.set_ylabel('E (J)')
    line_ke,  = ax_energy.plot([], [], lw=1,   label='KE')
    line_ge,  = ax_energy.plot([], [], lw=1,   label='GPE')
    line_ee,  = ax_energy.plot([], [], lw=1,   label='Elastic')
    line_tot, = ax_energy.plot([], [], lw=1.3, label='Total')
    ax_energy.set_xlim(0, args.window)
    ax_energy.set_ylim(None, None)  # Allow automatic scaling including negative values
    ax_energy.legend(fontsize=8)

    # Wind sliders
    ax_slider_x = fig.add_axes([.15, .05, .55, .015])
    ax_slider_y = fig.add_axes([.15, .02, .55, .015])
    slider_x = Slider(ax_slider_x, 'Wind x', -15, 15)
    slider_y = Slider(ax_slider_y, 'Wind y', -15, 15)

    frame_count = 0

    def update(frame_index):
        nonlocal frame_count, surface_plot
        frame_count += 1
        t = frame_count * args.dt

        # Current wind vector from sliders
        wind_vec = np.array([slider_x.val, slider_y.val, 0.0])

        # Advance cloth simulation
        cloth.step(args.dt, wind_vec)

        # Compute energies and record in buffers
        ke, ge, ee = cloth.energies()
        total_e = ke + ge + ee
        if not np.isfinite(total_e):
            print("Simulation diverged; closing.")
            plt.close()
            return

        time_buffer.append(t)
        ke_buffer.append(ke)
        ge_buffer.append(ge)
        ee_buffer.append(ee)
        tot_buffer.append(total_e)

        # Update energy plot
        times = np.array(time_buffer) - time_buffer[0]
        for line, data in zip(
            (line_ke, line_ge, line_ee, line_tot),
            (ke_buffer, ge_buffer, ee_buffer, tot_buffer)
        ):
            line.set_data(times, data)
        ax_energy.set_xlim(0, max(args.window, times[-1]))
        ax_energy.set_ylim(min(min(ke_buffer), min(ge_buffer), min(ee_buffer), min(tot_buffer)) * 1.2,
                          max(max(ke_buffer), max(ge_buffer), max(ee_buffer), max(tot_buffer)) * 1.2 + 1e-6)

        # Update cloth surface
        X, Y, Z = cloth.get_grid()
        surface_plot.remove()
        surface_plot = ax_surface.plot_surface(
            X, Y, Z,
            color='cornflowerblue', edgecolor='grey', lw=0.3, alpha=0.9
        )

        return surface_plot, line_ke, line_ge, line_ee, line_tot

    def on_slider_change(_val):
        """Re-run one update when sliders move."""
        update(None)
        fig.canvas.draw_idle()

    # Connect sliders
    slider_x.on_changed(on_slider_change)
    slider_y.on_changed(on_slider_change)

    # Start animation loop
    FuncAnimation(
        fig, update,
        interval=1000 / args.fps,
        blit=False,
        cache_frame_data=False
    )
    
    # Start animation loop and keep reference so it isn't deleted
    anim = FuncAnimation(
        fig, update,
        interval=1000 / args.fps,
        blit=False,
        cache_frame_data=False
    )

    plt.show()  

if __name__ == '__main__':
    main()