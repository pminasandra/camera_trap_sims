
import numpy as np
import vicsek
import vis

if __name__ == "__main__":
    N = 20
    # Block-diagonal adjacency: first 10 influence only first 10; second 10 only second 10
    A = np.zeros((N, N), dtype=bool)
    A[:10, :10] = True
    A[10:, 10:] = True
    # (Diagonal is True, so each agent includes itself.)

    traj, _ = vicsek.run_clustered_vicsek(
        N, 5_000,
        A=A,
        seed=42  # optional: reproducible
    )
    vis.animate_traj(traj, interval=20, show=True)
