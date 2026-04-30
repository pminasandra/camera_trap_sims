# Pranav Minasandra
# 17 Nov 2025
# pminasandra.github.io

import numpy as np
from typing import Optional, Tuple, Iterable
from collections.abc import Iterable as IterableObj

def run_cameras(
    cam_positions: np.ndarray,   # shape (M, 2)
    cam_radii: np.ndarray,       # shape (M,)
    positions: np.ndarray,       # shape (N, 2)
    *,
    assume_unique: bool = True
) -> np.ndarray:
    """
    For each individual, return which camera (if any) spots it.

    Parameters
    ----------
    cam_positions : (M, 2) array
        (x, y) locations of the M cameras.
    cam_radii : (M,) array
        Detection radius of each camera.
    positions : (N, 2) array
        (x, y) positions of the N individuals at a single timepoint.
    assume_unique : bool, default True
        If True, assumes an individual can be seen by at most one camera.
        If multiple cameras do see the same individual, the closest one
        is chosen.

    Returns
    -------
    spotted_by : (N,) array of int
        For individual i, spotted_by[i] is:
        - the index j of the camera that spots it (0 <= j < M), or
        - -1 if no camera spots it.
    """
    cam_positions = np.asarray(cam_positions, dtype=float)
    cam_radii = np.asarray(cam_radii, dtype=float)
    positions = np.asarray(positions, dtype=float)

    assert cam_positions.ndim == 2 and cam_positions.shape[1] == 2, "cam_positions must be (M,2)"
    assert cam_radii.ndim == 1 and cam_radii.shape[0] == cam_positions.shape[0], \
        "cam_radii must be (M,) matching cam_positions"
    assert positions.ndim == 2 and positions.shape[1] == 2, "positions must be (N,2)"

    N = positions.shape[0]
    M = cam_positions.shape[0]

    if M == 0:
        # No cameras: nobody is spotted
        return -np.ones(N, dtype=int)

    # Compute squared distances between individuals and cameras: (N, M)
    diff = positions[:, None, :] - cam_positions[None, :, :]  # (N, M, 2)
    dist2 = np.einsum("nmk,nmk->nm", diff, diff)              # (N, M)

    # Radius^2 threshold per camera
    r2 = cam_radii ** 2                                       # (M,)

    # Visibility mask: individual i is within camera j's radius?
    visible = dist2 <= r2[None, :]                            # (N, M)

    # Initialize all as "not seen"
    spotted_by = -np.ones(N, dtype=int)

    if assume_unique:
        # For each individual, choose the closest camera among those that see it
        # Mask out invisible cameras by setting distance to +inf
        dist2_masked = dist2.copy()
        dist2_masked[~visible] = np.inf

        # Closest camera index for each individual
        closest_idx = np.argmin(dist2_masked, axis=1)         # (N,)

        # Those that are actually seen by at least one camera
        seen_mask = np.any(visible, axis=1)                   # (N,)

        spotted_by[seen_mask] = closest_idx[seen_mask]
    else:
        # If you ever want to handle multi-camera visibility differently,
        # you can branch logic here.
        dist2_masked = dist2.copy()
        dist2_masked[~visible] = np.inf
        closest_idx = np.argmin(dist2_masked, axis=1)
        seen_mask = np.any(visible, axis=1)
        spotted_by[seen_mask] = closest_idx[seen_mask]

    return spotted_by

def make_camera_grid(
    k: int,
    radius: float | Iterable[float] = 0.5,
    box_size: float = 100.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set up k cameras on a sqrt(k) × sqrt(k) grid in a 100×100 arena (by default).

    Parameters
    ----------
    k : int
        Number of cameras. Must be a perfect square.
    radius : float or iterable, default 0.5
        Detection radius for each camera.
    box_size : float, default 100.0
        Size of the (square) arena; grid is built over [0, box_size] in x and y.
    offset_x : float, default 0.0
        Horizontal offset applied to all camera x-coordinates.
    offset_y : float, default 0.0
        Vertical offset applied to all camera y-coordinates.

    Returns
    -------
    cam_positions : (k, 2) array
        (x, y) positions of the cameras.
    cam_radii : (k,) array
        Detection radius for each camera (all identical).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    n_side = int(round(np.sqrt(k)))
    if n_side * n_side != k:
        raise ValueError(f"k must be a perfect square, got {k}")

    # Grid coordinates along one axis, centered in the box:
    # points at (0.5*box_size/n_side, 1.5*box_size/n_side, ..., (n_side-0.5)*box_size/n_side)
    coords_1d = (np.arange(n_side) + 0.5) * (box_size / n_side)

    xs, ys = np.meshgrid(coords_1d, coords_1d, indexing="xy")
    xs = xs.ravel() + offset_x
    ys = ys.ravel() + offset_y

    cam_positions = np.column_stack((xs, ys))   # (k, 2)
    if not isinstance(radius, IterableObj):
        cam_radii = np.full(k, radius, dtype=float)
    else:
        cam_radii = radius

    return cam_positions, cam_radii

def create_cameras(
    k: int,
    radius: float | Iterable = 0.5,
    box_size: float = 100.0,
    rng: np.random.Generator | None = None,
):
    """
    Create a sqrt(k) × sqrt(k) camera grid with a *random global offset*,
    guaranteed to keep all cameras strictly inside the arena.

    Parameters
    ----------
    k : int
        Number of cameras (must be a perfect square).
    radius : float, default 0.5
        Detection radius for each camera.
    box_size : float, default 100.0
        The size of the square arena, coordinates in [0, box_size].
    rng : np.random.Generator, optional
        Random generator for reproducible offsets.

    Returns
    -------
    cam_positions : (k,2) float array
    cam_radii     : (k,) float array
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Build centered grid (without offsets)
    cam_positions, cam_radii = make_camera_grid(
        k=k, radius=radius, box_size=box_size,
        offset_x=0.0, offset_y=0.0
    )

    # Step 2: Determine padding needed to keep cameras inside the arena
    # Cameras should not cross the boundaries [0, box_size].
    min_x = cam_positions[:, 0].min()
    max_x = cam_positions[:, 0].max()
    min_y = cam_positions[:, 1].min()
    max_y = cam_positions[:, 1].max()

    # Allowable offset ranges (so cameras always stay within bounds)
    offset_x_min = -min_x
    offset_x_max = box_size - max_x
    offset_y_min = -min_y
    offset_y_max = box_size - max_y

    # Step 3: Sample a random offset inside these ranges
    offset_x = rng.uniform(offset_x_min, offset_x_max)
    offset_y = rng.uniform(offset_y_min, offset_y_max)

    # Step 4: Apply offset
    cam_positions[:, 0] += offset_x
    cam_positions[:, 1] += offset_y

    return cam_positions, cam_radii


