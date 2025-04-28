"""
cloth_sim_rt.py (interactive)
Nathaniel Fargo
2025-04-17
------------------------------------------------------------
* Simulates a cloth object in real time
* Allows for user to adjust wind speed and direction
* Plots the cloth and the energy of the system over time
------------------------------------------------------------
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from collections import deque
from dataclasses import dataclass

# Constants
GRAVITY_ACCEL = 9.81
GRAVITY_VECTOR = np.array([0.0, 0.0, -GRAVITY_ACCEL])
DEFAULT_TIMESTEP = 0.02
DEFAULT_DAMPING = 0.01
DEFAULT_DRAG_COEFF = 0.3
DEFAULT_SPRING_K = 1000
DEFAULT_REST_LENGTH = 0.05
DEFAULT_MASS = 0.1

@dataclass
class SpringArray:
    """Stores spring endpoint indices, rest lengths, and spring constant."""
    idx_start: np.ndarray
    idx_end:   np.ndarray
    rest_length: np.ndarray
    k: float

class Cloth:
    def __init__(self, nx, nz, rest_length, mass, spring_k, damping, drag_coeff=0.5):
        # Grid dimensions and derived quantities
        self.nx = nx
        self.nz = nz
        self.num_points = nx * nz

        # Physical parameters
        self.mass       = mass
        self.spring_k   = spring_k
        self.damping    = damping
        self.drag_coeff = drag_coeff

        # Initial grid positions (x,z) and zero y
        x_positions = np.linspace(0, (nx - 1) * rest_length, nx)
        z_positions = np.linspace(0, - (nz - 1) * rest_length, nz)
        grid_x, grid_z = np.meshgrid(x_positions, z_positions)
        self.positions  = np.c_[grid_x.ravel(),             # x
                                 np.zeros(self.num_points),  # y
                                 grid_z.ravel()]             # z
        # Initialize with small random velocities
        VELOCITY_SCALE = 0.01
        self.velocities = np.random.randn(self.num_points, 3) * VELOCITY_SCALE

        # Fix the top row of points as anchors
        self.fixed_indices     = np.arange(nx)
        self.velocities[self.fixed_indices] = 0.0 # Ensure fixed points start at rest
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
            idx_start   = np.array(starts, dtype=int),
            idx_end     = np.array(ends,   dtype=int),
            rest_length = np.array(rests),
            k           = spring_k
        )

        # Record initial minimum z for GPE reference
        self.z_min = self.positions[:, 2].min()

        # Build triangular faces for computing normals
        faces = []
        for row in range(nz - 1):
            for col in range(nx - 1):
                base      = row * nx + col
                right     = base + 1
                down      = (row + 1) * nx + col
                down_right= down + 1
                faces.append([base, right, down])
                faces.append([right, down_right, down])
        self.faces = np.array(faces, dtype=int)

        # Adjacency: list of faces touching each node
        node_faces = [[] for _ in range(self.num_points)]
        for f_i, tri in enumerate(self.faces):
            for v in tri:
                node_faces[v].append(f_i)
        self.node_faces = [np.array(lst, dtype=int) for lst in node_faces]

    def _step(self, delta_t, wind_vector):
        """Perform a single integration step with gravity, springs, damping, and wind drag."""
        eps = 1e-12
        N   = self.num_points

        # 1) gravity
        forces = np.broadcast_to(self.mass * GRAVITY_VECTOR,
                                 self.positions.shape).copy()

        # 2) spring forces
        disp    = self.positions[self.springs.idx_end] - self.positions[self.springs.idx_start]
        lengths = np.linalg.norm(disp, axis=1, keepdims=True) + eps
        Fs      = (self.springs.k * (lengths - self.springs.rest_length[:, None])
                   * (disp / lengths))
        np.add.at(forces, self.springs.idx_start,  Fs)
        np.add.at(forces, self.springs.idx_end,   -Fs)

        # 3) linear damping
        forces += -self.damping * self.velocities

        # 4) aerodynamic drag + directivity
        # relative velocity to wind
        v_rel = wind_vector - self.velocities           # shape (N,3)
        speed = np.linalg.norm(v_rel, axis=1)           # shape (N,)
        unit  = v_rel / (speed[:, None] + eps)          # shape (N,3)

        # compute face normals
        p0, p1, p2 = (self.positions[self.faces[:, i]] for i in range(3))
        normals    = np.cross(p1 - p0, p2 - p0)
        normals   /= (np.linalg.norm(normals, axis=1)[:, None] + eps)

        # Calculate average relative velocity unit vector per face
        face_v_rel_unit = (unit[self.faces[:, 0]] + unit[self.faces[:, 1]] + unit[self.faces[:, 2]]) / 3.0
        # Normalize the average vector (important!)
        face_v_rel_unit /= (np.linalg.norm(face_v_rel_unit, axis=1)[:, None] + eps)

        # face directivity factor: 0.5 ↔ 1.0, using row-wise dot product
        # Use np.einsum for efficient row-wise dot product: sum(normals[k, i] * face_v_rel_unit[k, i] for i=0..2) for each face k
        face_dot_prod = np.einsum('ij,ij->i', normals, face_v_rel_unit) # shape (num_faces,)
        face_dir = 0.1 + 0.9 * np.abs(face_dot_prod)

        # node-wise average directivity
        dir_factors = np.array([
            face_dir[self.node_faces[i]].mean() if self.node_faces[i].size > 0 else 1.0
            for i in range(N)
        ])  # shape (N,)

        # drag magnitude ∝ v²
        drag_mag = self.drag_coeff * speed**2          # shape (N,)
        Fdrag    = unit * drag_mag[:, None] * dir_factors[:, None]
        forces  += Fdrag

        # 5) integrate
        self.velocities += (forces / self.mass) * delta_t
        self.positions  += self.velocities * delta_t

        # 6) re‑apply anchors
        self.positions[self.fixed_indices]  = self.initial_positions[self.fixed_indices]
        self.velocities[self.fixed_indices] = 0.0

    def step(self, dt, wind_vector):
        """
        Advance simulation by dt, splitting into stable sub-steps.
        """
        max_sub      = 0.15 * np.sqrt(self.mass / self.spring_k)
        num_substeps = max(1, int(np.ceil(dt / max_sub)))
        sub_dt       = dt / num_substeps
        for _ in range(num_substeps):
            self._step(sub_dt, wind_vector)

    def energies(self):
        """Return kinetic, gravitational, and elastic energies."""
        ke = 0.5 * self.mass * np.sum(self.velocities ** 2)
        ge = self.mass * GRAVITY_ACCEL * np.sum(
            self.positions[:, 2] - self.initial_positions[:, 2]
        )
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
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--nx',       type=int,   default=20)
    add('--nz',       type=int,   default=20)
    add('--rest_len', type=float, default=DEFAULT_REST_LENGTH)
    add('--mass',     type=float, default=DEFAULT_MASS)
    add('--k',        type=float, default=DEFAULT_SPRING_K)
    add('--damping',  type=float, default=DEFAULT_DAMPING)
    add('--drag_coeff', type=float, default=DEFAULT_DRAG_COEFF)
    add('--dt',       type=float, default=DEFAULT_TIMESTEP)
    add('--fps',      type=int,   default=45)
    add('--window',   type=float, default=10.0)
    args = parser.parse_args()

    cloth = Cloth(args.nx, args.nz, args.rest_len,
                  args.mass, args.k, args.damping,
                  args.drag_coeff)

    buffer_size = int(args.window / args.dt)
    time_buf, ke_buf, ge_buf, ee_buf, tot_buf = (
        deque(maxlen=buffer_size) for _ in range(5)
    )

    fig = plt.figure(figsize=(10, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)

    # Cloth subplot
    ax_surf = fig.add_subplot(gs[0], projection='3d')
    ax_surf.view_init(20, 30)
    for ax in (ax_surf.xaxis, ax_surf.yaxis, ax_surf.zaxis):
        ax.set_ticks([])
    
    span = max(args.nx - 1, args.nz - 1) * args.rest_len * 1.1
    xoffset = (args.nx - 1) * args.rest_len
    zoffset = (args.nz - 1) * args.rest_len
    ax_surf.set_xlim(-(span - xoffset) / 2, span - (span - xoffset) / 2)
    ax_surf.set_ylim(-span, span)
    ax_surf.set_zlim(-span, 0)
    ax_surf.set_xlabel('x'); ax_surf.set_ylabel('y'); ax_surf.set_zlabel('z')
    X, Y, Z = cloth.get_grid()
    surface_plot = ax_surf.plot_surface(
        X, Y, Z, color='cornflowerblue', edgecolor='none', alpha=0.9
    )

    # Energy subplot
    ax_e = fig.add_subplot(gs[1])
    ax_e.set_xlabel('t (s)'); ax_e.set_ylabel('E (J)')
    line_ke,  = ax_e.plot([], [], lw=1,   label='KE')
    line_ge,  = ax_e.plot([], [], lw=1,   label='GPE')
    line_ee,  = ax_e.plot([], [], lw=1,   label='Elastic')
    line_tot, = ax_e.plot([], [], lw=1.3, label='Total')
    ax_e.set_xlim(0, args.window)
    ax_e.set_ylim(None, None)
    ax_e.legend(fontsize=8)

    # Wind sliders
    ax_sx = fig.add_axes([.15, .05, .55, .015])
    ax_sy = fig.add_axes([.15, .02, .55, .015])
    wind_scale = 5
    slider_x = Slider(ax_sx, 'Wind x', -wind_scale, wind_scale, valinit=0)
    slider_y = Slider(ax_sy, 'Wind y', -wind_scale, wind_scale, valinit=0)

    frame_count = 0

    def update(_):
        nonlocal frame_count, surface_plot
        frame_count += 1
        t = frame_count * args.dt

        wind_vec = np.array([slider_x.val, slider_y.val, 0.0])
        cloth.step(args.dt, wind_vec)

        ke, ge, ee = cloth.energies()
        tot = ke + ge + ee
        if not np.isfinite(tot):
            print("Simulation diverged; closing.")
            plt.close(); return

        time_buf.append(t)
        ke_buf.append(ke); ge_buf.append(ge)
        ee_buf.append(ee); tot_buf.append(tot)

        times = np.array(time_buf) - time_buf[0]
        for line, data in zip(
            (line_ke, line_ge, line_ee, line_tot),
            (ke_buf, ge_buf, ee_buf, tot_buf)
        ):
            line.set_data(times, data)
        ax_e.set_xlim(0, max(args.window, times[-1]))
        mn = min(min(ke_buf), min(ge_buf), min(ee_buf), min(tot_buf)) * 1.2
        mx = max(max(ke_buf), max(ge_buf), max(ee_buf), max(tot_buf)) * 1.2 + 1e-6
        ax_e.set_ylim(mn, mx)

        X, Y, Z = cloth.get_grid()
        surface_plot.remove()
        surface_plot = ax_surf.plot_surface(
            X, Y, Z, color='cornflowerblue', edgecolor='none', alpha=0.9
        )

        return surface_plot, line_ke, line_ge, line_ee, line_tot

    slider_x.on_changed(lambda val: (update(None), fig.canvas.draw_idle()))
    slider_y.on_changed(lambda val: (update(None), fig.canvas.draw_idle()))

    anim = FuncAnimation(
        fig, update,
        interval=1000 / args.fps,
        blit=False,
        cache_frame_data=False
    )

    plt.show()

if __name__ == '__main__':
    main()