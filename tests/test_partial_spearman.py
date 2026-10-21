import numpy as np

from hohypemb.evaluation import partial_spearman


def test_partial_spearman_removes_a_common_cause():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z + 0.3 * rng.normal(size=500)
    y = z + 0.3 * rng.normal(size=500)
    rho, p = partial_spearman(x, y, z)
    assert abs(rho) < 0.1 and p > 0.01


def test_partial_spearman_keeps_a_direct_link():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    x = rng.normal(size=500)
    y = x + 0.2 * rng.normal(size=500)
    rho, p = partial_spearman(x, y, z)
    assert rho > 0.9 and p < 1e-10


def test_partial_spearman_degenerate_inputs():
    assert np.isnan(partial_spearman([1, 2, 3], [1, 2, 3], [1, 2, 3])[0])
    assert np.isnan(partial_spearman(np.ones(10), np.arange(10), np.arange(10))[0])
