import numpy as np

from restoration import g_exp, g_rat


def test_properties_of_both_conductances():
    s = np.linspace(0, 100, 50)
    for g in (g_exp, g_rat):
        v = g(s, 10.0)
        assert v[0] == 1.0                      # g(0) = 1: free diffusion in flat regions
        assert np.all(np.diff(v) <= 0)          # decreasing
        assert v[-1] < 0.05                     # large gradients are almost blocked
    assert np.isclose(g_exp(10, 10), np.exp(-1)) and np.isclose(g_rat(10, 10), 0.5)


def test_flux_peaks_at_the_threshold():
    """phi(s) = s g(s) increases then decreases; its peak is at K/sqrt(2) (exp) and K (rat)."""
    K = 10.0
    s = np.linspace(0, 40, 400001)
    assert abs(s[np.argmax(s * g_exp(s, K))] - K / np.sqrt(2)) < 1e-3
    assert abs(s[np.argmax(s * g_rat(s, K))] - K) < 1e-3
