"""Plain heat equation (conductance identically 1) = Gaussian blur with sigma = sqrt(2 * n * lam)."""
from .operators import neighbor_differences


def heat_step(u, lam=0.25):
    """One explicit step u + lam * (sum of the four neighbor differences) = u + lam * Laplacian(u)."""
    dN, dS, dE, dW = neighbor_differences(u)
    return u + lam * (dN + dS + dE + dW)


def heat_diffusion(u0, n_iter, lam=0.25, callback=None):
    """Run n_iter heat steps. callback(i, u) is called after each step (i = 1..n_iter)."""
    assert 0.0 < lam <= 0.25, "lambda must be in (0, 1/4] for stability"
    u = u0.astype(float).copy()
    for i in range(1, n_iter + 1):
        u = heat_step(u, lam)
        if callback is not None:
            callback(i, u)
    return u
