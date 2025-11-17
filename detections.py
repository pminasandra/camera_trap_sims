from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import cameras
import config


def run_cameras_on_trajectory(
    traj_path: str | Path,
    m: int,
    radius: float = 0.5,
    box_size: float = 100.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Given a saved Vicsek trajectory (.npy), generate a random camera setup
    and return a dataframe of detections.

    Parameters
    ----------
    traj_path : str or Path
        Path to a .npy file containing traj_pos with shape (T+1, N, 2).
    m : int
        Number of cameras.
    radius : float, default 0.5
        Camera detection radius.
    box_size : float, default 100.0
        Arena size (assumed square).
    seed : int, optional
        Seed for reproducible camera placement.

    Returns
    -------
    df : pandas.DataFrame
        Columns: ['timestamp', 'camera_id', 'individual'].
        One row per individual detected at each timestep.
    cam_positions : (m, 2) ndarray
        The camera positions used.
    cam_radii : (m,) ndarray
        The camera radii (all == radius).
    """
    traj_path = Path(traj_path)
    traj_pos = np.load(traj_path)   # (T+1, N, 2)
    if traj_pos.ndim != 3 or traj_pos.shape[2] != 2:
        raise ValueError("Expected traj_pos with shape (T+1, N, 2) in the npy file.")

    T1, N, _ = traj_pos.shape

    # Set up random cameras (but fixed for this whole run)
    rng = np.random.default_rng(seed)
    cam_positions, cam_radii = cameras.create_cameras(
        k=m,
        radius=radius,
        box_size=box_size,
        rng=rng,
    )

    records = []

    # Loop over timesteps, run cameras, collect detections
    for t in range(T1):
        positions_t = traj_pos[t]  # (N, 2)
        # Assuming signature: run_cameras(cam_positions, cam_radii, positions)
        spotted_by = cameras.run_cameras(cam_positions, cam_radii, positions_t)  # (N,)

        # For each individual seen by some camera, add a row
        for i in range(N):
            cam_id = int(spotted_by[i])
            if cam_id >= 0:  # -1 means not seen
                records.append(
                    {
                        "timestamp": t,     # timestep index
                        "camera_id": cam_id,
                        "individual": i,
                    }
                )

    df = pd.DataFrame.from_records(records, columns=["timestamp", "camera_id", "individual"])
    return df, cam_positions, cam_radii


if __name__ == "__main__":
    import os.path
    sightings, _, _ = run_cameras_on_trajectory(os.path.join(config.DATA,
        "trajectories/N10/separate/sim_000.npy"),
        m = 9)
    print(sightings)

