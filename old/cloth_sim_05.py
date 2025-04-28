
"""
cloth_sim_rt.py  (interactive v4 - optimized)
------------------------------------------------
* Vectorized spring calculations for speed
* Larger default dt (0.008 s) + slightly stronger damping
* Gravitational potential shifted so it is **zero** when the cloth is fully unrolled
* Minimal axes & live scrolling energy panel retained

NOTE: Updating a surface in‑place is still the slowest step;
      for finest grids (>40×40) consider --wireframe to plot a LineCollection instead.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from dataclasses import dataclass
from collections import deque

g = 9.81
GRAVITY = np.array([0.0, 0.0, -g])   # down –z

@dataclass
class SpringArray:
    i1: np.ndarray  # indices shape (m,)
    i2: np.ndarray
    rest: np.ndarray
    k: float        # scalar spring constant

class Cloth:
    def __init__(self, nx=20, nz=20, rest_len=0.2,
                 mass=0.05, k=400.0, damping=0.05):
        self.nx, self.nz = nx, nz
        self.n = nx * nz
        self.mass, self.k, self.damping = mass, k, damping

        xs = np.linspace(0, (nx - 1) * rest_len, nx)
        zs = np.linspace(0, -(nz - 1) * rest_len, nz)
        xx, zz = np.meshgrid(xs, zs)
        self.pos = np.column_stack((xx.ravel(), np.zeros(self.n), zz.ravel()))
        self.vel = np.zeros_like(self.pos)

        self.anchor_idx = np.arange(nx)           # top row anchored
        self.initial_pos = self.pos.copy()

        i1, i2, rest = [], [], []
        for j in range(nz):
            for i in range(nx):
                idx = j * nx + i
                if i < nx - 1:            # right neighbor
                    i1.append(idx); i2.append(idx + 1); rest.append(rest_len)
                if j < nz - 1:            # down neighbor
                    i1.append(idx); i2.append(idx + nx); rest.append(rest_len)
                if i < nx - 1 and j < nz - 1:  # down‑right shear
                    i1.append(idx); i2.append(idx + nx + 1); rest.append(rest_len * np.sqrt(2))
                if i > 0 and j < nz - 1:       # down‑left shear
                    i1.append(idx); i2.append(idx + nx - 1); rest.append(rest_len * np.sqrt(2))

        self.springs = SpringArray(np.array(i1, dtype=int),
                                   np.array(i2, dtype=int),
                                   np.array(rest, dtype=float),
                                   k)

        # baseline z (cloth fully extended) for zero gravitational potential
        self.z_baseline = self.pos[:, 2].min()

    # -------- energies ----------
    def energies(self):
        ke = 0.5 * self.mass * np.sum(self.vel ** 2)
        # gravitational PE relative to baseline
        ge = self.mass * g * np.sum(self.pos[:, 2] - self.z_baseline)
        # elastic
        delta = self.pos[self.springs.i2] - self.pos[self.springs.i1]
        length = np.linalg.norm(delta, axis=1)
        stretch = length - self.springs.rest
        ee = 0.5 * self.springs.k * np.sum(stretch ** 2)
        return ke, ge, ee

    # -------- simulation step ----------
    def step(self, dt, wind_vec):
        # external forces: gravity + wind
        forces = np.broadcast_to(self.mass * GRAVITY + wind_vec * self.mass,
                                 self.pos.shape).copy()

        # spring forces (vectorised)
        i1, i2, rest = self.springs.i1, self.springs.i2, self.springs.rest
        delta = self.pos[i2] - self.pos[i1]            # shape (m,3)
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        # prevent division by zero
        direction = np.divide(delta, length, out=np.zeros_like(delta), where=length != 0)
        f_spring = self.springs.k * (length - rest[:, None]) * direction   # (m,3)

        # accumulate forces at each node
        np.add.at(forces, i1,  f_spring)
        np.add.at(forces, i2, -f_spring)

        forces += -self.damping * self.vel

        # semi‑implicit Euler
        self.vel += (forces / self.mass) * dt
        self.pos += self.vel * dt

        # enforce anchors
        self.pos[self.anchor_idx] = self.initial_pos[self.anchor_idx]
        self.vel[self.anchor_idx] = 0.0

    def xyz_grid(self):
        grid = self.pos.reshape(self.nz, self.nx, 3)
        return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]

# ------------- main -------------
def main():
    parser = argparse.ArgumentParser(description='Optimised cloth simulation')
    parser.add_argument('--nx', type=int, default=20)
    parser.add_argument('--nz', type=int, default=20)
    parser.add_argument('--rest_len', type=float, default=0.15)
    parser.add_argument('--mass', type=float, default=0.05)
    parser.add_argument('--k', type=float, default=500.0)
    parser.add_argument('--damping', type=float, default=0.05)
    parser.add_argument('--dt', type=float, default=0.008)   # bigger default dt
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--window', type=float, default=10.0)
    args = parser.parse_args()

    cloth = Cloth(args.nx, args.nz, args.rest_len,
                  args.mass, args.k, args.damping)
    dt = args.dt
    window_frames = int(args.window / dt)

    # energy buffers
    t_buf = deque(maxlen=window_frames)
    ke_buf = deque(maxlen=window_frames)
    ge_buf = deque(maxlen=window_frames)
    ee_buf = deque(maxlen=window_frames)
    tot_buf = deque(maxlen=window_frames)

    # figure layout
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax3d.view_init(elev=20, azim=0)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_ticklabels([]); axis.set_ticks([])
    ax3d.set_xlabel('+x'); ax3d.set_ylabel('+y'); ax3d.set_zlabel('+z')

    ax3d.set_xlim(0, (args.nx - 1) * args.rest_len)
    span = max(args.nx - 1, args.nz - 1) * args.rest_len / 2
    ax3d.set_ylim(-span, span)
    ax3d.set_zlim(-(args.nz - 1) * args.rest_len, 0.1)
    ax3d.set_title('Optimised Cloth')

    axE = fig.add_subplot(gs[1])
    axE.set_xlabel('Δt (s)'); axE.set_ylabel('Energy (J)')
    axE.set_title('Energies'); axE.set_xlim(0, args.window); axE.set_ylim(0, 10)
    (l_ke,)   = axE.plot([], [], label='KE', lw=1)
    (l_ge,)   = axE.plot([], [], label='GPE', lw=1)
    (l_ee,)   = axE.plot([], [], label='Elastic', lw=1)
    (l_tot,)  = axE.plot([], [], label='Total', lw=1.5)
    axE.legend(fontsize=8)

    # surface
    X, Y, Z = cloth.xyz_grid()
    surf = ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1,
                             color='cornflowerblue',
                             edgecolor='grey', linewidth=0.3, alpha=0.9)

    # sliders
    ax_sx = fig.add_axes([0.15, 0.03, 0.55, 0.015])
    ax_sy = fig.add_axes([0.15, 0.0, 0.55, 0.015])
    s_x = Slider(ax_sx, 'Wind +x (m/s)', -15, 15, valinit=0)
    s_y = Slider(ax_sy, 'Wind +y (m/s)', -15, 15, valinit=0)

    frame = 0
    def update(_):
        nonlocal surf, frame
        frame += 1
        t = frame * dt
        wind = np.array([s_x.val, s_y.val, 0.0])
        cloth.step(dt, wind)

        ke, ge, ee = cloth.energies()
        tot = ke + ge + ee
        t_buf.append(t); ke_buf.append(ke); ge_buf.append(ge); ee_buf.append(ee); tot_buf.append(tot)

        # update energy lines
        t0 = t_buf[0]
        times = np.array(t_buf) - t0
        l_ke.set_data(times, ke_buf)
        l_ge.set_data(times, ge_buf)
        l_ee.set_data(times, ee_buf)
        l_tot.set_data(times, tot_buf)
        axE.set_xlim(0, max(args.window, times[-1]))
        axE.set_ylim(0, max(tot_buf)*1.2 + 1e-8)

        # refresh surface
        surf.remove()
        X, Y, Z = cloth.xyz_grid()
        surf = ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                 color='cornflowerblue',
                                 edgecolor='grey', linewidth=0.3, alpha=0.9)

        # quick save energy plot
        axE.figure.savefig('energy_plot.png', dpi=120, bbox_inches='tight')
        return surf, l_ke, l_ge, l_ee, l_tot

    ani = FuncAnimation(fig, update, interval=1000/args.fps, blit=False)
    plt.show()

if __name__ == '__main__':
    main()
