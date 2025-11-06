# vicsek.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union
from typing import Callable

ArrayLike = Union[np.ndarray, "np.typing.NDArray[np.floating]"]

def vicsek_step(
    positions: ArrayLike,          # shape (N, 2)
    headings: ArrayLike,           # shape (N,)
    v0: float = 0.03,              # speed per time step
    r: float = 1.0,                # interaction radius
    eta: float = 0.1,              # noise amplitude
    dt: float = 1.0,               # time step
    box_size: Optional[Union[float, Tuple[float, float]]] = None,
    A: Optional[np.ndarray] = None,# adjacency matrix (N×N, entries 0 or 1)
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Vicsek-model update step with reflecting boundaries
    and optional adjacency-matrix-defined interactions.

    Each A[i, j] = 1 means particle j influences particle i's heading.
    """
    pos = np.asarray(positions, dtype=float, order="C")
    th  = np.asarray(headings, dtype=float, order="C")
    N = pos.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    # Compute velocity vectors
    v = np.column_stack((np.cos(th), np.sin(th)))  # (N,2)

    # Pairwise distances (no periodic wrapping)
    diff = pos[:, None, :] - pos[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    neigh = dist2 <= (r * r)

    # Combine neighborhood and adjacency if A provided
    if A is not None:
        A = np.asarray(A, dtype=bool)
        assert A.shape == (N, N), "A must be N×N"
        neigh = neigh & A  # particle j affects i only if both close and A[i,j]==1

    # Compute local mean directions
    vx_sum = neigh @ v[:, 0]
    vy_sum = neigh @ v[:, 1]

    zero_mask = (vx_sum == 0.0) & (vy_sum == 0.0)
    vx_sum[zero_mask] = np.cos(th[zero_mask])
    vy_sum[zero_mask] = np.sin(th[zero_mask])
    mean_angle = np.arctan2(vy_sum, vx_sum)

    # Add angular noise
    th_new = mean_angle + rng.uniform(low=-eta/2, high=+eta/2, size=N)
    th_new = (th_new + np.pi) % (2 * np.pi) - np.pi

    # Update positions
    step = (v0 * dt) * np.column_stack((np.cos(th_new), np.sin(th_new)))
    pos_raw = pos + step

    # Reflecting boundary handling
    if box_size is not None:
        if np.isscalar(box_size):
            Lx = Ly = float(box_size)
        else:
            Lx, Ly = map(float, box_size)

        def reflect_and_flag(coord, L):
            twoL = 2 * L
            r = np.mod(coord, twoL)
            reflected = r > L
            coord_ref = np.where(reflected, twoL - r, r)
            return coord_ref, reflected

        x_ref, refl_x = reflect_and_flag(pos_raw[:, 0], Lx)
        y_ref, refl_y = reflect_and_flag(pos_raw[:, 1], Ly)
        pos_new = np.column_stack((x_ref, y_ref))
        th_new = np.where(refl_x, np.pi - th_new, th_new)
        th_new = np.where(refl_y, -th_new, th_new)
        th_new = (th_new + np.pi) % (2 * np.pi) - np.pi
    else:
        pos_new = pos_raw

    return pos_new, th_new

def simulate_vicsek(
    positions: ArrayLike,                 # (N, 2) initial positions
    headings: ArrayLike,                  # (N,) initial headings [radians]
    T: int,                               # number of time steps to simulate
    *,
    v0: float = 0.03,
    r: float = 1.0,
    eta: float = 0.1,
    dt: float = 1.0,
    box_size: Optional[Union[float, Tuple[float, float]]] = None,
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,
    A: Optional[np.ndarray]=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a Vicsek simulation (reflecting boundaries if box_size is given).

    Parameters
    ----------
    positions : (N,2) array
        Initial particle positions.
    headings : (N,) array
        Initial particle headings in radians.
    T : int
        Number of steps to run.
    v0, r, eta, dt, box_size
        Passed through to vicsek_step (reflecting BC if box_size is provided).
    seed : int, optional
        Seed for reproducible randomness.
    progress : callable(step, positions, headings) -> None, optional
        Called after each completed step (1..T). Useful for live logging/plots.

    Returns
    -------
    traj_positions : (T+1, N, 2) array
        Positions at times 0..T (includes the initial state at index 0).
    traj_headings  : (T+1, N) array
        Headings at times 0..T (includes the initial state at index 0).
    """
    pos0 = np.asarray(positions, dtype=float)
    th0  = np.asarray(headings, dtype=float)
    assert pos0.ndim == 2 and pos0.shape[1] == 2, "positions must be (N,2)"
    assert th0.ndim == 1 and th0.shape[0] == pos0.shape[0], "headings must be (N,) matching positions"
    assert isinstance(T, int) and T >= 0, "T must be a non-negative integer"

    N = pos0.shape[0]
    rng = np.random.default_rng(seed)

    traj_pos = np.empty((T + 1, N, 2), dtype=float)
    traj_th  = np.empty((T + 1, N), dtype=float)
    traj_pos[0] = pos0
    traj_th[0]  = th0

    pos, th = pos0.copy(), th0.copy()
    for t in range(1, T + 1):
        pos, th = vicsek_step(
            pos, th,
            v0=v0, r=r, eta=eta, dt=dt,
            box_size=box_size,
            rng=rng,
            A=A
        )
        traj_pos[t] = pos
        traj_th[t]  = th
        if progress is not None:
            progress(t, pos, th)

    return traj_pos, traj_th

def run_clustered_vicsek(
    N: int,
    T: int,
    *,
    box_size: Union[float, Tuple[float, float]] = 100.0,  # 100×100 arena by default
    v0: float = 0.5,
    r: float = 2.0,
    eta: float = 0.2,
    dt: float = 1.0,
    cluster_sigma: Optional[float] = None,  # stddev of initial cluster
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,
    A: Optional[np.ndarray]=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize a Vicsek group in close proximity and simulate for T steps.

    Parameters
    ----------
    N : int
        Number of individuals.
    T : int
        Number of time steps to simulate.
    box_size : float or (Lx, Ly), default 100.0
        Arena size; reflecting boundaries are applied during the simulation.
    v0, r, eta, dt
        Standard Vicsek parameters (speed, radius, noise amplitude, time step).
    cluster_sigma : float, optional
        Standard deviation (in position units) of the initial tight cluster.
        If None, defaults to max(0.25 * r, 0.5).
    seed : int, optional
        Random seed for reproducibility.
    progress : callable(step, positions, headings) -> None, optional
        Called after each step (1..T).

    Returns
    -------
    traj_positions : (T+1, N, 2)
    traj_headings  : (T+1, N)
    """
    assert isinstance(N, int) and N > 0, "N must be a positive integer"
    assert isinstance(T, int) and T >= 0, "T must be a non-negative integer"

    rng = np.random.default_rng(seed)

    # Resolve box size
    if np.isscalar(box_size):
        Lx = Ly = float(box_size)
    else:
        Lx, Ly = map(float, box_size)

    # Choose a cluster spread small relative to interaction radius
    if cluster_sigma is None:
        cluster_sigma = max(0.25 * r, 0.5)

    # Keep the initial cluster well inside the arena margins
    margin_x = 4.0 * cluster_sigma
    margin_y = 4.0 * cluster_sigma
    cx = rng.uniform(margin_x, max(margin_x, Lx - margin_x))
    cy = rng.uniform(margin_y, max(margin_y, Ly - margin_y))

    # Sample initial positions as a tight Gaussian blob around (cx, cy)
    pos0 = np.empty((N, 2), dtype=float)
    pos0[:, 0] = rng.normal(loc=cx, scale=cluster_sigma, size=N)
    pos0[:, 1] = rng.normal(loc=cy, scale=cluster_sigma, size=N)
    # Ensure inside bounds (so first step doesn't bounce immediately)
    pos0[:, 0] = np.clip(pos0[:, 0], 0.0, Lx)
    pos0[:, 1] = np.clip(pos0[:, 1], 0.0, Ly)

    # Random initial headings in (-pi, pi]
    th0 = rng.uniform(low=-np.pi, high=np.pi, size=N)

    # Run the simulation with reflecting boundaries
    traj_pos, traj_th = simulate_vicsek(
        positions=pos0,
        headings=th0,
        T=T,
        v0=v0,
        r=r,
        eta=eta,
        dt=dt,
        box_size=(Lx, Ly),
        seed=seed,
        progress=progress,
        A=A
    )
    return traj_pos, traj_th

