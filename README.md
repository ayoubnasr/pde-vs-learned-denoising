# Image restoration from scratch — Phase 1: Perona–Malik anisotropic diffusion

A small, tested NumPy implementation of Perona–Malik denoising (Perona & Malik, 1990), compared against plain heat-equation smoothing (Gaussian blur) on grayscale images with additive Gaussian noise.

**Status:** Phase 1 only (classical diffusion). Later phases (TV/ROF, BM3D baseline, small CNN) are planned and **not implemented yet**.

![Camera image, sigma 25: clean, noisy, heat equation, Perona–Malik](results/comparison.png)

## Results

Mean over 4 bundled scikit-image test images (camera, coins, moon, astronaut-as-grayscale), Gaussian noise, fixed seed, pixel range 0–255. `K` is the single value that maximises the mean PSNR across the images; the number of iterations is chosen per image.

| noise σ | method | K | mean iters | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| 15 | noisy input | – | – | 24.60 | 0.446 |
| 15 | heat equation | – | 4 | 29.27 | 0.752 |
| 15 | Perona–Malik, exponential g | 40 | 4 | 31.98 | 0.831 |
| 15 | Perona–Malik, rational g | 10 | 12 | 32.34 | 0.857 |
| 25 | noisy input | – | – | 20.16 | 0.289 |
| 25 | heat equation | – | 8 | 27.66 | 0.655 |
| 25 | Perona–Malik, exponential g | 60 | 6 | 29.62 | 0.761 |
| 25 | Perona–Malik, rational g | 20 | 11 | 29.99 | 0.784 |
| 50 | noisy input | – | – | 14.14 | 0.142 |
| 50 | heat equation | – | 20 | 25.44 | 0.557 |
| 50 | Perona–Malik, exponential g | 120 | 13 | 26.72 | 0.640 |
| 50 | Perona–Malik, rational g | 30 | 23 | 27.16 | 0.699 |

Full per-image numbers: [`results/sweep.csv`](results/sweep.csv). More figures: [`results/k_sweep.png`](results/k_sweep.png), [`results/psnr_curves.png`](results/psnr_curves.png), [`results/edge_demo.png`](results/edge_demo.png).

**How to read these numbers (caveats).**
- The best iteration (and `K`) are picked by comparing to the *clean* image, so these are **oracle-tuned** scores: what each method can reach when tuned well, not what a blind user would get.
- Four small images and one noise seed per setting. Not the standard benchmarks (Set12 / BSD68); to run on those, see below.
- Only grayscale, i.i.d. Gaussian noise.

## The method in one paragraph

Heat-equation smoothing blurs edges as much as noise. Perona–Malik makes the diffusion strength depend on the local gradient,

  ∂ₜu = div( g(|∇u|) ∇u ),  g(s) = exp(−(s/K)²)  or  1 / (1 + (s/K)²),

so smoothing is strong in flat regions (g ≈ 1) and weak across strong edges (g ≈ 0). `K` is the gradient size that separates "smooth this" from "keep this".

## Implementation notes

Explicit four-neighbor scheme in **flux form** (`src/restoration/perona_malik.py`):

  uⁿ⁺¹ = u + λ [ c_N ∇_N u + c_S ∇_S u + c_E ∇_E u + c_W ∇_W u ],  c_D = g(|∇_D u|, K)

where ∇_D u is the difference to the neighbor in direction D. Consequences, each covered by a test:
- Every update is a weighted average of the pixel and its four neighbors when 0 < λ ≤ ¼ (the center weight is 1 − λ·Σc_D ≥ 1 − 4λ). Hence **no new maxima or minima**, and the code rejects λ outside (0, ¼].
- Edge replication at the border means no flux crosses it, so **total brightness is conserved**.
- With g ≡ 1 the scheme is the heat equation, i.e. a Gaussian blur with σ = √(2·n·λ).

## Known limitation

Noise also creates steep local gradients. When the noise is strong relative to `K`, Perona–Malik treats some noise as edges and leaves isolated speckles in flat regions (visible in the zoomed crop above). This is a property of the method; a standard remedy is to compute the gradient on a pre-smoothed copy (Catté et al., 1992), which is not implemented here.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                   # 23 tests
python experiments/run_sweep.py             # writes results/sweep.csv (a few minutes)
python experiments/summarize.py             # results/summary.md
python experiments/make_figures.py          # comparison, K sweep, PSNR curves
python experiments/edge_demo.py             # 1D edge-preservation figure
python experiments/run_sweep.py --images-dir path/to/Set12 --out results/set12.csv   # your own images
```

```python
from restoration import load_image, add_gaussian_noise, perona_malik, g_rat, psnr

clean = load_image("camera")
noisy = add_gaussian_noise(clean, sigma=25, seed=0)
out = perona_malik(noisy, n_iter=10, K=20.0, g=g_rat)      # lam defaults to 0.25
print(psnr(clean, noisy), psnr(clean, out))
```

## Layout

```
src/restoration/
  data.py          test images, folder loader, noise
  metrics.py       PSNR, SSIM
  operators.py     neighbor differences (edge replication)
  conductance.py   g_exp, g_rat
  heat.py          heat-equation baseline
  perona_malik.py  Perona–Malik step and loop
  evaluation.py    best-iteration scoring (oracle)
tests/             properties: conservation, max principle, heat = Gaussian, edge preservation
experiments/       sweep, summary, figures
results/           CSV, summary table, figures
```

## References

- P. Perona, J. Malik, *Scale-space and edge detection using anisotropic diffusion*, IEEE TPAMI 12(7), 1990.
- F. Catté, P.-L. Lions, J.-M. Morel, T. Coll, *Image selective smoothing and edge detection by nonlinear diffusion*, SIAM J. Numer. Anal. 29(1), 1992.
