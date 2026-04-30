# Pranav Minasandra
# 17 Nov 2025
# pminasandra.github.io

from __future__ import annotations
import multiprocessing as mp
import os

import numpy as np

import config
import vicsek


def make_adjacency(N: int, condition: str) -> np.ndarray:
    """
    Create adjacency matrix A (N×N, dtype=bool) for a given condition.
    """
    A = np.zeros((N, N), dtype=bool)

    if condition == "together":
        A[:, :] = True                          # everyone influences everyone
    elif condition == "subgroups":
        half = N // 2
        A[:half, :half] = True
        A[half:, half:] = True
    elif condition == "separate":
        np.fill_diagonal(A, True)               # only self-influence
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return A

population_sizes = list(range(10, 101, 10))
conditions = ["together", "subgroups", "separate"]
n_reps = 100

# Parameters for Vicsek simulation (adjust as needed)
T = 5000
box_size = 100.0

base_dir = os.path.join(config.DATA, "trajectories")

def make_traj(N, cond):
    cond_dir = os.path.join(base_dir, f"N{N}", cond)
    os.makedirs(cond_dir, exist_ok=True)

    for rep in range(n_reps):
        A = make_adjacency(N, cond)
        seed = rep  # deterministic reproducibility

        traj_pos, _ = vicsek.run_clustered_vicsek(
            N=N,
            T=T,
            A=A,
            box_size=box_size,
            seed=seed,
        )

        # File name: sim_000.npy, sim_001.npy, etc.
        out_path = os.path.join(cond_dir, f"sim_{rep:03d}.npy")
        np.save(out_path, traj_pos)

        print(f"Saved {out_path}")

if __name__ == "__main__":
    os.makedirs(base_dir, exist_ok=True)

    tgts = []
    for N in population_sizes:
        for cond in conditions:
            tgts.append((N, cond))

    pool = mp.Pool()
    pool.starmap(make_traj, tgts)
    pool.close()
    pool.join()


