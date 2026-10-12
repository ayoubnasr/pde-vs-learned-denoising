import numpy as np

from restoration import neighbor_differences


def test_constant_image_has_zero_differences():
    c = np.full((8, 8), 7.0)
    assert all(np.all(d == 0) for d in neighbor_differences(c))


def test_ramp_signs_and_border_replication():
    ramp = np.tile(np.arange(8.0), (8, 1))           # brightness grows to the east
    dN, dS, dE, dW = neighbor_differences(ramp)
    assert np.all(dE[:, :-1] == 1) and np.all(dW[:, 1:] == -1) and np.all(dN == 0) and np.all(dS == 0)
    assert np.all(dE[:, -1] == 0) and np.all(dW[:, 0] == 0)   # nothing flows across the border


def test_sum_of_differences_is_the_laplacian_stencil():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(10, 10))
    s = sum(neighbor_differences(u))
    p = np.pad(u, 1, mode="edge")
    stencil = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * u
    assert np.allclose(s, stencil)
