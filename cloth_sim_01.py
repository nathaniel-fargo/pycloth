
"""
cloth_sim.py
-------------
Mass-spring cloth simulation with energy tracking and optional animation.

Usage
-----
python cloth_sim.py --steps 2000 --dt 0.005 --animate

Command-line arguments
----------------------
--nx, --ny      Grid dimensions (default 15 x 15)
--rest_len      Rest length between adjacent nodes (m)
--mass          Mass of each node (kg)
--k             Spring constant for structural & shear springs (N/m)
--damping       Linear damping coefficient (kg/s)
--dt            Time step (s)
--steps         Number of integration steps
--animate       If given, saves `cloth_animation.mp4` (requires ffmpeg)
--out_prefix    Prefix for generated output files
"""

import argparse
import csv
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from pathlib import Path

GRAVITY = np.array([0.0, -9.81, 0.0])  # m/s^2, negative y direction

@dataclass
class Spring:
    i1: int
    i2: int
    rest: float
    k: float

class Cloth:
    def __init__(self, nx: int, ny: int, rest_len: float, mass: float,
                 k: float, damping: float, dt: float):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.mass = mass
        self.k = k
        self.damping = damping
        self.dt = dt

        # Initial positions (flat rectangle in x–z plane, y = 0)
        xs = np.linspace(0, (nx - 1) * rest_len, nx)
        zs = np.linspace(0, (ny - 1) * rest_len, ny)
        xx, zz = np.meshgrid(xs, zs)
        self.pos = np.column_stack((xx.ravel(), np.zeros(self.n), zz.ravel()))
        self.vel = np.zeros_like(self.pos)

        # Anchor the top row (ny = 0) so it stays fixed
        self.anchor_idx = [i for i in range(nx)]
        self.initial_pos = self.pos.copy()

        # Build springs (structural + shear)
        self.springs = []
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                # right neighbor
                if i < nx - 1:
                    self._add_spring(idx, idx + 1, rest_len)
                # down neighbor
                if j < ny - 1:
                    self._add_spring(idx, idx + nx, rest_len)
                # shear: down-right
                if i < nx - 1 and j < ny - 1:
                    self._add_spring(idx, idx + nx + 1, rest_len * np.sqrt(2))
                # shear: down-left
                if i > 0 and j < ny - 1:
                    self._add_spring(idx, idx + nx - 1, rest_len * np.sqrt(2))

    def _add_spring(self, i1, i2, rest):
        self.springs.append(Spring(i1, i2, rest, self.k))

    def step(self):
        forces = np.zeros_like(self.pos)
        # Gravity
        forces += self.mass * GRAVITY

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
        self.vel += (forces / self.mass) * self.dt
        self.pos += self.vel * self.dt

        # Re‑anchor
        self.pos[self.anchor_idx] = self.initial_pos[self.anchor_idx]
        self.vel[self.anchor_idx] = 0.0

    def energies(self):
        # Kinetic
        ke = 0.5 * self.mass * np.sum(np.square(self.vel))
        # Gravitational
        ge = -self.mass * GRAVITY[1] * np.sum(self.pos[:, 1])
        # Elastic
        ee = 0.0
        for s in self.springs:
            delta = self.pos[s.i2] - self.pos[s.i1]
            length = np.linalg.norm(delta)
            ee += 0.5 * s.k * (length - s.rest) ** 2
        return ke, ge, ee

def simulate(args):
    cloth = Cloth(args.nx, args.ny, args.rest_len, args.mass, args.k,
                  args.damping, args.dt)
    energies = []
    for step in range(args.steps):
        cloth.step()
        energies.append(cloth.energies())
        if step % 100 == 0:
            print(f"Step {step}/{args.steps}\r", end="", file=sys.stderr)

    energies = np.array(energies)
    out_prefix = args.out_prefix
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Save energies to CSV
    with open(f"{out_prefix}_energies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kinetic", "gravitational", "elastic"])
        writer.writerows(energies)

    # Plot energies
    plt.figure()
    plt.plot(energies[:, 0], label="kinetic")
    plt.plot(energies[:, 1], label="gravitational")
    plt.plot(energies[:, 2], label="elastic")
    plt.plot(energies.sum(axis=1), label="total")
    plt.xlabel("Step")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_energies_plot.png", dpi=300)

    # Optional animation
    if args.animate:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(0, (args.nx - 1) * args.rest_len)
        ax.set_ylim(-(args.ny * args.rest_len), 1)
        ax.set_zlim(0, (args.ny - 1) * args.rest_len)
        scat = ax.scatter([], [], [], s=10)

        def update(frame):
            cloth.step()
            xyz = cloth.pos
            scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=args.steps,
                                      interval=20, blit=False)
        ani.save(f"{out_prefix}_animation.mp4", writer='ffmpeg', fps=30)

def main():
    parser = argparse.ArgumentParser(description="Mass–spring cloth simulation")
    parser.add_argument('--nx', type=int, default=15)
    parser.add_argument('--ny', type=int, default=15)
    parser.add_argument('--rest_len', type=float, default=0.2)
    parser.add_argument('--mass', type=float, default=0.05)
    parser.add_argument('--k', type=float, default=400.0)
    parser.add_argument('--damping', type=float, default=0.02)
    parser.add_argument('--dt', type=float, default=0.004)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--out_prefix', type=str, default='cloth')
    args = parser.parse_args()
    simulate(args)

if __name__ == '__main__':
    main()
