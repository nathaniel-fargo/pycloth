
"""
cloth_sim_rt.py
----------------
Real-time interactive cloth simulation with wind slider.

Run:
    python cloth_sim_rt.py

Requirements:
    numpy, matplotlib (with 3-D backend), tqdm (optional for CLI progress).

Controls:
    * Horizontal slider at bottom changes wind speed (m/s) blowing +x direction.
    * Close the window to stop the simulation and save a frame-grabbing MP4 if desired.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from dataclasses import dataclass

GRAVITY = np.array([0.0, -9.81, 0.0])  # Down in −y

@dataclass
class Spring:
    i1: int
    i2: int
    rest: float
    k: float

class Cloth:
    def __init__(self, nx=15, ny=15, rest_len=0.2, mass=0.05, k=400.0, damping=0.02):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.mass = mass
        self.k = k
        self.damping = damping

        # Initial grid in x–y plane (z = 0) with top row anchored (y = 0)
        xs = np.linspace(0, (nx - 1) * rest_len, nx)
        ys = np.linspace(0, -(ny - 1) * rest_len, ny)  # downward is negative y
        xx, yy = np.meshgrid(xs, ys)
        self.pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(self.n)))
        self.vel = np.zeros_like(self.pos)

        self.anchor_idx = [i for i in range(nx)]  # top row
        self.initial_pos = self.pos.copy()

        # Build springs (structural + shear)
        self.springs = []
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if i < nx - 1:      # right neighbor
                    self._add_spring(idx, idx + 1, rest_len)
                if j < ny - 1:      # down neighbor
                    self._add_spring(idx, idx + nx, rest_len)
                if i < nx - 1 and j < ny - 1:  # down-right shear
                    self._add_spring(idx, idx + nx + 1, rest_len * np.sqrt(2))
                if i > 0 and j < ny - 1:       # down-left shear
                    self._add_spring(idx, idx + nx - 1, rest_len * np.sqrt(2))

    def _add_spring(self, i1, i2, rest):
        self.springs.append(Spring(i1, i2, rest, self.k))

    def step(self, dt, wind_speed):
        forces = np.zeros_like(self.pos)
        # Gravity
        forces += self.mass * GRAVITY

        # Wind (simple constant horizontal force along +x)
        wind_force = np.array([wind_speed * self.mass, 0.0, 0.0])
        forces += wind_force

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

        # Semi‑implicit Euler
        self.vel += (forces / self.mass) * dt
        self.pos += self.vel * dt

        # Re‑anchor
        self.pos[self.anchor_idx] = self.initial_pos[self.anchor_idx]
        self.vel[self.anchor_idx] = 0.0

def run():
    # Parameters
    nx, ny = 15, 15
    rest_len = 0.15
    mass = 0.05
    k = 500.0
    damping = 0.03
    dt = 0.004

    cloth = Cloth(nx, ny, rest_len, mass, k, damping)

    # Matplotlib setup
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=90)
    ax.set_xlim(0, (nx - 1) * rest_len)
    ax.set_ylim(-(ny - 1) * rest_len, 0.1)
    ax.set_zlim(-((nx + ny) * rest_len) / 4, ((nx + ny) * rest_len) / 4)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Interactive Cloth Simulation')

    scat = ax.scatter(cloth.pos[:, 0], cloth.pos[:, 1], cloth.pos[:, 2], s=12, c='tab:blue')

    # Slider for wind speed
    slider_ax = fig.add_axes([0.20, 0.03, 0.60, 0.025])
    wind_slider = Slider(slider_ax, 'Wind (m/s)', -15.0, 15.0, valinit=0.0)

    def update(frame):
        wind = wind_slider.val
        cloth.step(dt, wind)
        xyz = cloth.pos
        scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return scat,

    ani = FuncAnimation(fig, update, interval=16, blit=False)  # ~60 FPS

    plt.show()

if __name__ == '__main__':
    run()
