import numpy as np

from restoration import add_gaussian_noise, best_iteration, heat_diffusion, load_image, psnr


def test_best_iteration_is_the_argmax_and_beats_noisy():
    clean = load_image()[:128, :128]
    noisy = add_gaussian_noise(clean, 25, seed=0)
    run = lambda n, cb: heat_diffusion(noisy, n, callback=cb)
    p, s, it, img = best_iteration(run, clean, 40)
    scores = []
    heat_diffusion(noisy, 40, callback=lambda i, u: scores.append(psnr(clean, u)))
    assert it == int(np.argmax(scores)) + 1 and np.isclose(p, max(scores))
    assert p > psnr(clean, noisy) and 0.0 < s < 1.0
