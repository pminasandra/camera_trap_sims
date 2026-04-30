# generate_sightings.py
from __future__ import annotations
from typing import Iterable, Tuple

import os
from pathlib import Path
import numpy as np
import pandas as pd

import config
import cameras
from detections import run_cameras_on_trajectory


CAMSETS = {}
for s in np.arange(2,10)**2:
    cam_radii = np.full(s, 0.3) + 0.7*np.random.random(s)# uniform from 0.3 to 1.0
    pos, rad = cameras.make_camera_grid(s, cam_radii)
    scams = (pos, rad)
    CAMSETS[s] = scams

def generate_all_sightings(
    *,
    cams: Tuple | None = None,
    camera_sizes=None,
    radius: float | Iterable = 0.5,
    box_size: float = 100.0,
    seed_base: int = 1234,
):
    """
    Loop over all saved trajectories in config.DATA/trajectories,
    and generate sightings for a variety of camera counts.

    For each trajectory sim_XXX.npy:
        - generates sightings using camera_sizes = k cameras
        - saves sightings under:
              config.DATA/sightings/Nxx/<condition>/sim_XXX_ncams_k.parquet
        - saves camera positions under:
              config.DATA/cameras/Nxx/<condition>/sim_XXX_ncams_k.parquet

    Parameters
    ----------
    cams: dict, pre-made camera positions and radii. If None, automatically generates
        camera locations.
    camera_sizes : list[int], optional
        List of camera-grid sizes (must be perfect squares).
        Default = [4, 9, 16, 25, 36, 49, 64, 81].
    radius : float or iterable, default 0.5
        Camera detection radius.
    box_size : float, default 100.0
        Arena width/height.
    seed_base : int
        Base seed; actual seed = seed_base + rep_idx
    """

    if camera_sizes is None:
        camera_sizes = [4, 9, 16, 25, 36, 49, 64, 81]

    traj_root = Path(config.DATA) / "trajectories"
    sightings_root = Path(config.DATA) / "sightings"
    cameras_root = Path(config.DATA) / "cameras"
    if cams is not None:
        cameras_root.mkdir(parents=True, exist_ok=True)
        for s in camera_sizes:
            file = np.hstack((cams[s][0], cams[s][1][..., None]))
            fname = cameras_root / f"global_ncams_{s}.parquet"
            pd.DataFrame(file, columns=["x", "y",
                                        "r"]).to_parquet(fname)
            print("saved", fname)

    for N_dir in traj_root.iterdir():
        if not N_dir.is_dir():
            continue

        for cond_dir in N_dir.iterdir():
            if not cond_dir.is_dir():
                continue

            # Prepare parallel directories
            sight_dir = sightings_root / N_dir.name / cond_dir.name
            cam_dir = cameras_root / N_dir.name / cond_dir.name
            sight_dir.mkdir(parents=True, exist_ok=True)
            cam_dir.mkdir(parents=True, exist_ok=True)



            # Loop over all sim_XXX.npy files
            for npy_file in sorted(cond_dir.glob("sim_*.npy")):

                # E.g. sim_000.npy → base name "sim_000"
                base = npy_file.stem

                # Choose a reproducible seed for camera placement
                # per simulation
                seed_for_this_sim = seed_base + int(base.split("_")[1])

                for k in camera_sizes:

                    # === Run camera sightings ===
                    if cams is None:
                        df_sight, cam_positions, _ = run_cameras_on_trajectory(
                            traj_path=npy_file,
                            m=k,
                            radius=radius,
                            box_size=box_size,
                            seed=seed_for_this_sim,
                        )
                    else:
                        df_sight, _, _ = run_cameras_on_trajectory(
                            traj_path=npy_file,
                            cams=cams[k]
                        )

                    # === Save sightings ===
                    sight_out = sight_dir / f"{base}_ncams_{k}.parquet"
                    df_sight.to_parquet(sight_out, index=False)
                    print(f"Saved: {sight_out}")

                    # === Save camera positions ===
                    if cams is None:
                        cam_out = cam_dir / f"{base}_ncams_{k}.parquet"
                        pd.DataFrame(cam_positions, columns=["x", "y"]).to_parquet(cam_out, index=False)
                        print(f"Saved: {cam_out}")

if __name__ == "__main__":
    generate_all_sightings(cams=CAMSETS)
