# nanotron-quant

Quantitative research primitives, organized by what they protect against.

| Subpackage | What it solves |
|------------|----------------|
| `cv/`       | **Leakage in time-series CV.** Walk-forward, expanding-window, and López de Prado purged + embargoed K-fold (with the combinatorial variant for PBO). |
| `labels/`   | **Sample-label horizon overlap.** Triple-barrier and meta-labeling. |
| `features/` | **Stationarity vs. memory.** Fixed-width fractional differentiation + an `optimal_d` helper that finds the minimum `d` for which an ADF test rejects. |

All public APIs are pickle-clean and operate on either pandas or numpy.
No mutable globals. Sklearn-compatible signatures (`split(X)`,
`get_n_splits(X)`) where it makes sense.

## Install

```bash
pip install -e .[dev]
```

`optimal_d` additionally needs `statsmodels`:

```bash
pip install statsmodels
```

## Test

```bash
pytest -q
```

## Why these specific primitives

These three modules cover the three most common ways a financial-ML
backtest accidentally tells you the future:

1. **Time-series CV that ignores label horizons.** Random K-fold leaks
   information through any label whose realization window crosses the
   train/test boundary. `PurgedKFold` removes those overlapping training
   events; `embargo_pct` removes a buffer immediately after each test
   fold to defeat residual serial correlation.
2. **Labeling that cheats by using fixed forward returns.** Triple-barrier
   gives every event the same volatility-scaled risk/reward profile
   instead of a fixed look-ahead window — closer to what a real bet
   would experience.
3. **Differencing that destroys memory.** Plain first differences
   throw away the level information the model needs. Fractional
   differentiation gives you stationarity at the smallest cost in memory,
   and `optimal_d` picks that smallest cost automatically.

Used by the backtest harness in `quant/backtest/` and the supervised-model
training pipelines in `ml/`.
