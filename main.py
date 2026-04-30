# Pranav Minasandra
# pminasandra.github.io
# Dec 5 2025

import multiprocessing as mp

import numpy as np

import config
import trajectories
import generate_sightings

if __name__ == "__main__":
    print("Creating trajectories. (This might take a while)")
    os.makedirs(base_dir, exist_ok=True)

    tgts = []
    for N in trajectories.population_sizes:
        for cond in trajectories.conditions:
            tgts.append((N, cond))

    pool = mp.Pool()
    pool.starmap(trajectories.make_traj, tgts)
    pool.close()
    pool.join()

    print("Generating pseudocameratrap datasets.")
    generate_sightings.generate_all_sightings(cams=generate_sightings.CAMSETS)
