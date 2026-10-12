"""Conductance functions g(s) of Perona and Malik (1990). s = |gradient| >= 0, K = edge threshold.

Both are non-negative, decreasing, g(0) = 1. The flux phi(s) = s * g(s) increases up to
s = K/sqrt(2) (exponential) or s = K (rational) and decreases after: gradients below that
are smoothed, gradients above it are protected (and sharpened).
"""
import numpy as np


def g_exp(s, K):
    """g(s) = exp(-(s/K)^2)."""
    return np.exp(-(s / K) ** 2)


def g_rat(s, K):
    """g(s) = 1 / (1 + (s/K)^2)."""
    return 1.0 / (1.0 + (s / K) ** 2)


CONDUCTANCES = {"exp": g_exp, "rat": g_rat}
