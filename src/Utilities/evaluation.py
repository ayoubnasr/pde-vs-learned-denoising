"""Helpers for scoring an iterative denoiser against the clean image."""
from .metrics import psnr, ssim


def best_iteration(run, clean, n_max):
    """Run an iterative method and report the best iteration, judged against the clean image.

    run(n_max, callback) must call callback(i, u) after each step i = 1..n_max.
    Returns (best_psnr, best_ssim, best_iter, best_image). Because the best iteration is chosen
    by looking at the clean image, this is an ORACLE: it measures what each method can reach with
    tuned settings, not what it achieves when tuned blindly.
    """
    best = {"psnr": float("-inf"), "iter": 0, "img": None}

    def cb(i, u):
        p = psnr(clean, u)
        if p > best["psnr"]:
            best.update(psnr=p, iter=i, img=u.copy())

    run(n_max, cb)
    return best["psnr"], ssim(clean, best["img"]), best["iter"], best["img"]
