"""Control-plane API for project-nanotron.

The data plane (Rust + axum) handles real-time websocket streaming from
shared memory.  This package owns the *non-real-time* surface: strategy
lifecycle, risk-limit configuration, backtest dispatch, and snapshot
inspection.  Keeping those off the hot path lets us iterate on
operator-facing semantics (auth, validation, audit) without touching
the Rust binary.
"""

__version__ = "0.1.0"

from .app import create_app  # noqa: E402

__all__ = ["create_app", "__version__"]
