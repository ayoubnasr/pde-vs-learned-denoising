"""Discrete differential operators on the pixel grid (spacing h = 1)."""
import numpy as np


def neighbor_differences(u):
    """Differences from each pixel to its N, S, E, W neighbor (neighbor minus center).

    Borders use edge replication, so the difference across the border is 0: no brightness
    flows across the image boundary, which makes the total brightness conserved.
    """
    p = np.pad(u, 1, mode="edge")
    dN = p[:-2, 1:-1] - u
    dS = p[2:, 1:-1] - u
    dE = p[1:-1, 2:] - u
    dW = p[1:-1, :-2] - u
    return dN, dS, dE, dW
