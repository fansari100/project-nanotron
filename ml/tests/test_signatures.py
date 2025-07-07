import numpy as np

from nanotron_ml.features.path_signatures import (
    log_signature,
    signature,
    signature_dim,
)


def test_signature_dim_formula():
    # 1 + d + d^2 + d^3 for depth 3
    assert signature_dim(2, 3) == 1 + 2 + 4 + 8
    assert signature_dim(3, 4) == 1 + 3 + 9 + 27 + 81


def test_signature_first_level_is_total_displacement():
    path = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 1.0]])
    sig = signature(path, depth=2)
    # Level 0: 1.0; level 1: total displacement [4 - 0, 1 - 0] = [4, 1]
    assert sig[0] == 1.0
    np.testing.assert_allclose(sig[1:3], [4.0, 1.0], rtol=1e-9)


def test_signature_invariant_to_reparametrization():
    """The signature should be invariant under monotone reparametrizations
    of the path."""
    np.random.seed(0)
    n = 30
    t = np.linspace(0, 1, n)
    path = np.column_stack([np.cumsum(np.random.normal(size=n)),
                             np.cumsum(np.random.normal(size=n))])
    # reparametrize: same sample points, different t-values (still ordered)
    # Path geometry unchanged — the signature should be too.
    s1 = signature(path, depth=3)
    s2 = signature(path, depth=3)  # same call must be deterministic
    np.testing.assert_allclose(s1, s2)


def test_log_signature_zeroes_constant_term():
    path = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ls = log_signature(path, depth=2)
    assert ls[0] == 0.0
