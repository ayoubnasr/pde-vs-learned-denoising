"""Image-quality metrics. Always compare against the CLEAN image, with the right data range."""
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, test, data_range=255.0):
    """PSNR in dB: 10 log10(data_range^2 / MSE)."""
    return peak_signal_noise_ratio(ref, test, data_range=data_range)


def ssim(ref, test, data_range=255.0):
    """Structural similarity index (scikit-image defaults)."""
    return structural_similarity(ref, test, data_range=data_range)
