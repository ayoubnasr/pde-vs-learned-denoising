import numpy as np
import pytest

from restoration import add_gaussian_noise, load_image, load_test_images, psnr, ssim


def test_images_are_float_grayscale_0_255():
    for name, img in load_test_images().items():
        assert img.ndim == 2 and img.dtype == np.float64, name
        assert 0.0 <= img.min() and img.max() <= 255.0, name


def test_unknown_image_raises():
    with pytest.raises(ValueError):
        load_image("nope")


def test_noise_level_and_reproducibility():
    clean = load_image()
    a = add_gaussian_noise(clean, 25, seed=0)
    b = add_gaussian_noise(clean, 25, seed=0)
    assert np.array_equal(a, b)
    assert abs((a - clean).std() - 25) < 0.5


def test_psnr_matches_formula_and_ssim_of_identical_is_one():
    clean = load_image()
    noisy = add_gaussian_noise(clean, 25, seed=0)
    assert abs(psnr(clean, noisy) - 20 * np.log10(255 / 25)) < 0.1
    assert ssim(clean, clean) == pytest.approx(1.0)
