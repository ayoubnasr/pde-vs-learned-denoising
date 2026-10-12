"""Perona-Malik anisotropic diffusion, explicit 4-neighbor flux-form scheme.

    u_new = u + lam * [ c_N dN + c_S dS + c_E dE + c_W dW ],   c_D = g(|d_D|, K)

Each conductance is computed from the SAME difference it multiplies (the flux across that link).
In this form the weights on the neighbors are lam*c_D >= 0 and the weight on the center is
1 - lam*sum(c_D) >= 0 whenever lam <= 1/4 and g <= 1, so the update is a weighted average:
no new maxima or minima, and total brightness is conserved (the fluxes telescope).
"""
import numpy as np

from .conductance import g_exp
from .operators import neighbor_differences


def pm_step(u, K, lam=0.25, g=g_exp):
    """One explicit Perona-Malik step."""
    dN, dS, dE, dW = neighbor_differences(u)
    return u + lam * (g(np.abs(dN), K) * dN + g(np.abs(dS), K) * dS
                      + g(np.abs(dE), K) * dE + g(np.abs(dW), K) * dW)


def perona_malik(u0, n_iter, K, lam=0.25, g=g_exp, callback=None):
    """Run n_iter Perona-Malik steps. callback(i, u) is called after each step (i = 1..n_iter).

    K is on the same scale as the pixel values (0-255 here). lam must be in (0, 1/4].
    """
    assert 0.0 < lam <= 0.25, "lambda must be in (0, 1/4] for stability"
    u = u0.astype(float).copy()
    for i in range(1, n_iter + 1):
        u = pm_step(u, K, lam, g)
        if callback is not None:
            callback(i, u)
    return u
