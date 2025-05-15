"""Quantitative research primitives for project-nanotron.

Three top-level subpackages:
    cv/         time-aware cross-validation (walk-forward, purged, embargoed)
    labels/     event-based labels (triple-barrier, meta-labeling)
    features/   leakage-safe transforms (fractional differentiation, vol scaling)

Designed to be pickle-clean, sklearn-compatible where it makes sense, and
to play well with both pandas (Series/DataFrame indexed by datetime) and
numpy (when speed matters).  No mutable global state.
"""

__version__ = "0.1.0"
