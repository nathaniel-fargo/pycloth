"""
cloth_sim_rt.py   (interactive v2)
---------------------------------
Real‑time cloth simulation with a surface mesh, resolution options,
and two wind sliders (along +x and +y).

Example:
    python cloth_sim_rt.py --nx 25 --nz 25 --rest_len 0.12
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from dataclasses import dataclass

GRAVITY = np.array([0.0, 0.0, -9.81])   # down = –z

@dataclass
class Spring:
    i1: int
    i2: int
    rest: float
    k: float

class Cloth:
    """Mass–spring cloth model."""
    def __init__(self, nx=15, nz=15, rest_len=0.2, mass=0.05, k=400.0, damping=0.02):
        self.nx, self.nz = nx, nz
        self.n = nx * nz
        self.mass, self.k, self.damping = mass, k, damping

        # Build grid in x–z plane (y = 0)
        xs = np.linspace(0, (nx - 1) * rest_len, nx)
        zs = np.linspace(0, -(nz - 1) * rest_len, nz)   # downward negative
        xx, zz = np.meshgrid(xs, zs)
        self.pos = np.column_stack((xx.ravel(), np.zeros(self.n), zz.ravel()))
        self.vel = np.zeros_like(self.pos)

        # anchor top row
        self.anchor_idx = [i for i in range(nx)]
        self.initial_pos = self.pos.copy()

        self.springs = []
        for j in range(nz):
            for i in range(nx):
                idx = j * nx + i
                if i < nx - 1:      # right
                    self._add_spring(idx, idx + 1, rest_len)
                if j < nz - 1:      # down
                    self._add_spring(idx, idx + nx, rest_len)
                if i < nx - 1 and j < nz - 1:    # down‑right shear
                    self._add_spring(idx, idx + nx + 1, rest_len * np.sqrt(2))
                if i > 0 and j < nz - 1:         # down‑left shear
                    self._add_spring(idx, idx + nx - 1, rest_len * np.sqrt(2))

    def _add_spring(self, i1, i2, rest):
        self.springs.append(Spring(i1, i2, rest, self.k))

    def step(self, dt, wind_vec):
        """Advance simulation by dt (semi‑implicit Euler)."""
        forces = np.zeros_like(self.pos)
        forces += self.mass * GRAVITY                       # gravity
        forces += wind_vec * self.mass                      # wind per node

        # Springs
        for s in self.springs:
            delta = self.pos[s.i2] - self.pos[s.i1]
            length = np.linalg.norm(delta)
            if length == 0:
                continue
            direction = delta / length
            f = s.k * (length - s.rest) * direction
            forces[s.i1] += f
            forces[s.i2] -= f

        # Damping
        forces += -self.damping * self.vel

        # Integrate
        self.vel += (forces / self.mass) * dt
        self.pos += self.vel * dt

        # Re‑anchor
        self.pos[self.anchor_idx] = self.initial_pos[self.anchor_idx]
        self.vel[self.anchor_idx] = 0.0

    # helpers for visualization
    def xyz_grid(self):
        """Return x, y, z arrays shaped (nz, nx) for surface plotting."""
        grid = self.pos.reshape(self.nz, self.nx, 3)
        return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]

def main():
    parser = argparse.ArgumentParser(description="Interactive cloth simulation")
    parser.add_argument('--nx', type=int, default=15, help='grid points in x')
    parser.add_argument('--nz', type=int, default=15, help='grid points in z')
    parser.add_argument('--rest_len', type=float, default=0.15)
    parser.add_argument('--mass', type=float, default=0.05)
    parser.add_argument('--k', type=float, default=500.0)
    parser.add_argument('--damping', type=float, default=0.03)
    parser.add_argument('--dt', type=float, default=0.004)
    parser.add_argument('--fps', type=int, default=60)
    args = parser.parse_args()

    cloth = Cloth(args.nx, args.nz, args.rest_len, args.mass, args.k, args.damping)
    dt = args.dt

    # Matplotlib setup
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=0)
    ax.set_xlim(0, (args.nx - 1) * args.rest_len)
    span = max((args.nx - 1), (args.nz - 1)) * args.rest_len / 2
    ax.set_ylim(-span, span)
    ax.set_zlim(-(args.nz - 1) * args.rest_len, 0.1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Interactive Cloth Simulation (surface mesh)')

    # Initial surface
    X, Y, Z = cloth.xyz_grid()
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='cornflowerblue',
                           edgecolor='grey', linewidth=0.5, alpha=0.9)

    # Wind sliders
    slider_ax_x = fig.add_axes([0.18, 0.04, 0.6, 0.02])
    slider_ax_y = fig.add_axes([0.18, 0.0, 0.6, 0.02])
    sx = Slider(slider_ax_x, 'Wind X (m/s)', -15.0, 15.0, valinit=0.0)
    sy = Slider(slider_ax_y, 'Wind Y (m/s)', -15.0, 15.0, valinit=0.0)

    def update(frame):
        wind_vec = np.array([sx.val, sy.val, 0.0])
        cloth.step(dt, wind_vec)

        nonlocal surf
        surf.remove()                       # remove old surface
        X, Y, Z = cloth.xyz_grid()
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='cornflowerblue',
                               edgecolor='grey', linewidth=0.5, alpha=0.9)
        return surf,

    interval_ms = 1000 / args.fps
    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)
    plt.show()

if __name__ == '__main__':
    main()
