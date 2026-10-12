import numpy as np
import pytest

from restoration import (add_gaussian_noise, g_exp, g_rat, heat_step, load_image,
                         perona_malik, pm_step)


@pytest.fixture(scope="module")
def noisy():
    return add_gaussian_noise(load_image(), 25, seed=0)


def test_unit_conductance_is_the_heat_equation(noisy):
    ones = lambda s, K: np.ones_like(s)
    a = perona_malik(noisy, 30, K=1.0, g=ones)
    b = noisy.copy()
    for _ in range(30):
        b = heat_step(b)
    assert np.allclose(a, b)


@pytest.mark.parametrize("g", [g_exp, g_rat])
def test_conservation_and_discrete_maximum_principle(noisy, g):
    out = perona_malik(noisy, 30, K=20.0, g=g)
    assert np.isclose(out.mean(), noisy.mean())                       # total brightness conserved
    assert out.min() >= noisy.min() - 1e-9 and out.max() <= noisy.max() + 1e-9   # no new extremes


def test_constant_image_is_a_fixed_point():
    const = np.full((16, 16), 5.0)
    assert np.allclose(perona_malik(const, 10, K=3.0), const)


def test_each_step_stays_inside_local_neighbourhood_range(noisy):
    """Weighted-average property: every new pixel lies within [min, max] of itself and its 4 neighbours."""
    u = noisy[:64, :64]
    v = pm_step(u, 20.0, 0.25, g_rat)
    p = np.pad(u, 1, mode="edge")
    stack = np.stack([u, p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    assert np.all(v <= stack.max(axis=0) + 1e-9) and np.all(v >= stack.min(axis=0) - 1e-9)


@pytest.mark.parametrize("lam", [0.0, -0.1, 0.26, 0.5])
def test_lambda_outside_stability_range_is_rejected(noisy, lam):
    with pytest.raises(AssertionError):
        perona_malik(noisy, 1, K=20.0, lam=lam)


def edge_width(img):
    """Width in pixels of the transition of a vertical step edge, from the row-averaged profile."""
    prof = img.mean(axis=0)
    lo, hi = prof[:20].mean(), prof[-20:].mean()
    t = (prof - lo) / (hi - lo)
    return int(np.sum((t > 0.2) & (t < 0.8)))


def test_edges_stay_sharp_while_flat_regions_are_smoothed():
    step = np.full((64, 128), 40.0)
    step[:, 64:] = 160.0
    noisy_step = add_gaussian_noise(step, 8, seed=0)
    h = noisy_step.copy()
    for _ in range(40):
        h = heat_step(h)
    p = perona_malik(noisy_step, 40, K=30.0, g=g_rat)
    assert edge_width(p) < edge_width(h)                              # heat smears the edge, PM does not
    assert p[:, :40].std() < 0.5 * noisy_step[:, :40].std()           # but PM still removes noise


def test_exponential_and_rational_both_denoise(noisy):
    clean = load_image()
    from restoration import psnr
    for g, K in ((g_exp, 20.0), (g_rat, 20.0)):
        out = perona_malik(noisy, 15, K=K, g=g)
        assert psnr(clean, out) > psnr(clean, noisy) + 3.0
