"""nanotron-data — market, crypto, on-chain and alt-data connectors.

Layout::

    connectors/      vendor-specific clients (polygon, alpaca, …)
        base.py      retry + circuit-breaker primitives every connector uses
    features/        polars/duckdb feature pipeline (see Sep 18 commit)
    quality/         great_expectations integration (see Sep 23 commit)

All connectors share a typed protocol::

    class BarsConnector(Protocol):
        async def bars(symbol, start, end, frequency) -> pd.DataFrame: ...

So callers can swap vendors without changing downstream code.  The
retry + circuit-breaker layer in `base.py` is non-negotiable: every
public network call goes through it.
"""

__version__ = "0.1.0"
