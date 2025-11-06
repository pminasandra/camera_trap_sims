# vis.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple

def animate_traj(
    traj_pos: np.ndarray,                 # shape (T+1, N, 2)
    tail: int = 100,                      # how many past steps to show as a trail
    interval: int = 40,                   # ms between frames
    figsize: Tuple[float, float] = (6, 6),
    colors: Optional[np.ndarray] = None,  # optional (N,) or (N,3) colors
    axis_limits: Optional[Tuple[float, float, float, float]] = None,  # (xmin, xmax, ymin, ymax)
    show: bool = True,
):
    """
    Animate a Vicsek trajectory:
      - Lines: last `tail` time steps for each particle (if available)
      - Scatter: current positions

    Parameters
    ----------
    traj_pos : (T+1, N, 2) array
        Positions over time (time-major).
    tail : int, default 100
        Number of past steps to display as the trailing line.
    interval : int, default 40
        Delay between frames in milliseconds.
    figsize : tuple, default (6,6)
        Figure size passed to matplotlib.
    colors : array-like, optional
        Either (N,) of color spec or (N,3)/(N,4) RGB(A) values for particles.
    axis_limits : (xmin, xmax, ymin, ymax), optional
        If not provided, computed from data with a small padding.
    show : bool, default True
        If True, calls plt.show() before returning.

    Returns
    -------
    fig, anim
        Matplotlib figure and the FuncAnimation object.
    """
    assert traj_pos.ndim == 3 and traj_pos.shape[2] == 2, "traj_pos must be (T+1, N, 2)"
    T1, N, _ = traj_pos.shape
    assert T1 >= 1 and N >= 1

    # Build/validate colors
    if colors is None:
        # default: a simple colormap by index
        cmap = plt.get_cmap("tab20", N)
        colors = np.array([cmap(i) for i in range(N)])
    else:
        colors = np.asarray(colors)
        if colors.ndim == 1:  # list of color strings
            pass
        else:
            assert colors.shape[0] == N, "colors must have N rows"

    # Axes limits
    if axis_limits is None:
        xmin = np.nanmin(traj_pos[..., 0])
        xmax = np.nanmax(traj_pos[..., 0])
        ymin = np.nanmin(traj_pos[..., 1])
        ymax = np.nanmax(traj_pos[..., 1])
        pad_x = 0.05 * max(1e-9, xmax - xmin)
        pad_y = 0.05 * max(1e-9, ymax - ymin)
        axis_limits = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    # Figure and artists
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_limits[0], axis_limits[1])
    ax.set_ylim(axis_limits[2], axis_limits[3])

    # Create a Line2D for each particle's trail
    line_artists = []
    for i in range(N):
        color_i = colors[i] if colors.ndim > 1 else colors[i]
        (ln,) = ax.plot([], [], lw=0.5, alpha=0.5, color=color_i)
        line_artists.append(ln)

    # Scatter for current positions
    scat = ax.scatter([0]*N, [0]*N, s=2, c=colors if colors.ndim > 1 else None)

    # Frame updater
    def init():
        for ln in line_artists:
            ln.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        return (*line_artists, scat)

    def update(frame):
        # frame goes from 0 .. T
        t0 = max(0, frame - tail)
        # Update each trail
        for i, ln in enumerate(line_artists):
            xs = traj_pos[t0:frame + 1, i, 0]
            ys = traj_pos[t0:frame + 1, i, 1]
            ln.set_data(xs, ys)
        # Update scatter to current positions
        scat.set_offsets(traj_pos[frame, :, :])
        return (*line_artists, scat)

    anim = FuncAnimation(
        fig,
        update,
        frames=range(T1),
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    if show:
        plt.show()

    return fig, anim

