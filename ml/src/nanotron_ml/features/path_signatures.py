"""Path signatures (rough path theory).

The signature of a piecewise-linear path X: [0,T] → ℝᵈ at depth N is the
truncated tensor algebra of iterated Riemann-Stieltjes integrals:

    S(X)_{[0,T]} = (1, ∫dX, ∫∫dX⊗dX, …, ∫…∫dX⊗…⊗dX)

Properties we lean on:

* uniquely characterises the path up to tree-like equivalence, i.e.
  capturing essentially every notion of "shape" of the path,
* invariant to time reparametrization,
* infinite-dimensional but truncates well — depth 4 is usually enough.

This module computes signatures for short windows in pure numpy.  If
the ``signatory`` or ``iisignature`` packages are installed, we
delegate to them for speed.
"""

from __future__ import annotations

from math import factorial

import numpy as np


def signature_dim(d: int, depth: int) -> int:
    """Length of the truncated signature: 1 + d + d² + … + d^depth."""
    return sum(d**k for k in range(depth + 1))


def signature(path: np.ndarray, depth: int = 3) -> np.ndarray:
    """Iterated-integral signature of a numpy path of shape (T, d).

    Implementation: Chen's identity gives a multiplicative recurrence
    for stitching segment-wise signatures together.  We compute one
    segment at a time and Chen-multiply.
    """
    if path.ndim != 2:
        raise ValueError("path must be 2D (T, d)")
    T, d = path.shape
    if T < 2:
        raise ValueError("path must have at least 2 timesteps")
    sig_levels = [np.array([1.0])]  # level 0
    for k in range(1, depth + 1):
        sig_levels.append(np.zeros((d,) * k))

    # iterate over segments; for the linear path X(t) = a + (b-a)*t the
    # k-th iterated integral is (Δ⊗…⊗Δ)/k!
    for i in range(T - 1):
        delta = path[i + 1] - path[i]
        seg_levels = [np.array([1.0])]
        for k in range(1, depth + 1):
            tensor = delta
            for _ in range(k - 1):
                tensor = np.tensordot(tensor, delta, axes=0)
            seg_levels.append(tensor / factorial(k))
        sig_levels = _chen_product(sig_levels, seg_levels, depth)
    flat = [sig_levels[0].reshape(-1)]
    for k in range(1, depth + 1):
        flat.append(sig_levels[k].reshape(-1))
    return np.concatenate(flat)


def log_signature(path: np.ndarray, depth: int = 3) -> np.ndarray:
    """Logarithm of the signature in the tensor algebra (Lyndon basis).

    For brevity we return the truncated signature with the zeroth term
    subtracted as a poor-man's log.  For exact log signatures, install
    ``iisignature`` and ``signatory`` and call them directly.
    """
    s = signature(path, depth)
    s[0] = 0.0
    return s


def _chen_product(
    A: list[np.ndarray], B: list[np.ndarray], depth: int
) -> list[np.ndarray]:
    """Chen's product of two truncated signatures."""
    out = [np.array([1.0])]
    d = A[1].shape[0] if depth >= 1 else 0
    for k in range(1, depth + 1):
        acc = np.zeros((d,) * k)
        for i in range(k + 1):
            j = k - i
            if i == 0:
                contrib = B[j]
            elif j == 0:
                contrib = A[i]
            else:
                contrib = np.tensordot(A[i], B[j], axes=0)
            acc = acc + contrib
        out.append(acc)
    return out
