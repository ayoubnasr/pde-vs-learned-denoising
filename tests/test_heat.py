import numpy as np
from scipy.ndimage import gaussian_filter

from restoration import add_gaussian_noise, heat_diffusion, load_image


def test_heat_matches_gaussian_blur():
    """n steps of size lam equal a Gaussian blur with sigma = sqrt(2 n lam) (checked away from borders)."""
    noisy = add_gaussian_noise(load_image(), 25, seed=0)
    n, lam = 40, 0.25
    h = heat_diffusion(noisy, n, lam)
    ref = gaussian_filter(noisy, np.sqrt(2 * n * lam), mode="nearest")
    inner = (slice(30, -30), slice(30, -30))
    rms_diff = np.sqrt(np.mean((h - ref)[inner] ** 2))
    rms_change = np.sqrt(np.mean((h - noisy)[inner] ** 2))
    assert rms_diff < 0.1 * rms_change


def test_heat_conserves_mean():
    noisy = add_gaussian_noise(load_image(), 25, seed=0)
    assert np.isclose(heat_diffusion(noisy, 40).mean(), noisy.mean())
