"""Test images and synthetic noise. All images are grayscale float64 on the 0-255 scale."""
from pathlib import Path

import numpy as np
from skimage import color, data, io

# Bundled with scikit-image (no download needed).
TEST_IMAGES = ("camera", "coins", "moon", "astronaut")


def load_image(name="camera"):
    """Return a clean grayscale test image as float64 in [0, 255]."""
    if name not in TEST_IMAGES:
        raise ValueError(f"unknown image {name!r}; choose from {TEST_IMAGES}")
    img = getattr(data, name)()
    if img.ndim == 3:                       # astronaut is RGB
        img = color.rgb2gray(img) * 255.0
    return np.asarray(img, dtype=np.float64)


def load_test_images(names=TEST_IMAGES):
    """Dict {name: image} for the bundled test images."""
    return {n: load_image(n) for n in names}


def load_folder(path, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    """Load every image in a folder (e.g. Set12 or BSD68) as grayscale float64 in [0, 255]."""
    out = {}
    for f in sorted(Path(path).iterdir()):
        if f.suffix.lower() not in extensions:
            continue
        img = io.imread(f)
        if img.ndim == 3:
            img = color.rgb2gray(img[..., :3]) * 255.0
        elif img.dtype != np.uint8 and img.max() <= 1.0:
            img = img * 255.0
        out[f.stem] = np.asarray(img, dtype=np.float64)
    if not out:
        raise FileNotFoundError(f"no images found in {path}")
    return out


def add_gaussian_noise(img, sigma, seed=0):
    """Add i.i.d. Gaussian noise N(0, sigma^2) to every pixel. No clipping, fixed seed."""
    rng = np.random.default_rng(seed)
    return img + rng.normal(0.0, sigma, img.shape)
