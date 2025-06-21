import pandas as pd
import polars as pl
import pytest

from nanotron_data.features import (
    DuckDBFeatures,
    FeatureRegistry,
    FeatureSpec,
    add_returns,
    cross_sectional_zscore,
    realized_vol,
    rolling_features,
    vol_target,
)


def _toy_bars(n=50):
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    rows = []
    for sym in ("A", "B"):
        for i, t in enumerate(ts):
            rows.append(
                {
                    "timestamp": t,
                    "symbol": sym,
                    "close": 100.0 + i * (1.0 if sym == "A" else 0.5),
                }
            )
    return pl.from_pandas(pd.DataFrame(rows))


def test_add_returns_first_row_is_null_per_group():
    df = _toy_bars()
    out = add_returns(df)
    nulls = out.group_by("symbol").agg(pl.col("ret").is_null().sum())
    assert (nulls["ret"] == 1).all()


def test_realized_vol_window_finite():
    df = add_returns(_toy_bars())
    out = realized_vol(df, window=10)
    rv = out["rv"].drop_nulls()
    assert rv.is_finite().all()


def test_rolling_features_emits_expected_columns():
    df = add_returns(_toy_bars())
    out = rolling_features(df, col="ret", windows=(5, 20))
    for c in ("ret_m5", "ret_s5", "ret_z5", "ret_m20", "ret_s20", "ret_z20"):
        assert c in out.columns


def test_cross_sectional_zscore_zero_mean_per_timestamp():
    df = add_returns(_toy_bars())
    out = cross_sectional_zscore(df, "ret")
    means = (
        out.drop_nulls("ret_xz").group_by("timestamp").agg(pl.col("ret_xz").mean()).to_pandas()
    )
    # mean of a 2-element zscore sample is exactly zero
    assert (means["ret_xz"].abs() < 1e-9).all()


def test_vol_target_scales_inversely_to_vol():
    df = add_returns(_toy_bars())
    df = realized_vol(df, window=10)
    df = df.with_columns(pl.col("ret").alias("signal"))
    out = vol_target(df, "signal", "rv", target_vol=0.10)
    df_pd = out.drop_nulls(["signal_vt", "rv"]).to_pandas()
    rebuilt = df_pd["signal"] * (0.10 / df_pd["rv"])
    assert (df_pd["signal_vt"] - rebuilt).abs().max() < 1e-9


def test_duckdb_realized_vol_sql_matches_polars(tmp_path):
    df = _toy_bars().to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    parquet_path = tmp_path / "bars.parquet"
    df.to_parquet(parquet_path)
    duck = DuckDBFeatures()
    duck.register_parquet("bars", str(parquet_path))
    out = duck.query(duck.realized_vol_sql("bars", window=10))
    assert "rv" in out.columns


def test_feature_registry_round_trip(tmp_path):
    reg = FeatureRegistry(root=tmp_path)
    spec = FeatureSpec(name="momo_5d", version="v1", description="5-day momentum")
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "symbol": ["A", "A", "A"],
            "value": [0.01, 0.02, -0.01],
        }
    )
    reg.write(spec, df)
    out = reg.read("momo_5d", version="v1")
    assert (out["value"].to_list() == [0.01, 0.02, -0.01])


def test_feature_registry_pit_lookup_no_lookahead(tmp_path):
    reg = FeatureRegistry(root=tmp_path)
    spec = FeatureSpec(name="momo_5d", version="v1")
    feature_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-05", "2024-01-10"]
            ),
            "symbol": ["A", "A", "A"],
            "value": [0.10, 0.20, 0.30],
        }
    )
    reg.write(spec, feature_df)
    when = pd.Series(
        pd.to_datetime(["2024-01-03", "2024-01-07", "2024-01-15"]),
        index=pd.Index(["A", "A", "A"], name="symbol"),
    )
    out = reg.lookup("momo_5d", when, by="symbol")
    # No lookahead: at 2024-01-03 we should see 0.10 (set on 2024-01-01),
    # at 2024-01-07 we should see 0.20, etc.
    assert out.iloc[0] == 0.10
    assert out.iloc[1] == 0.20
    assert out.iloc[2] == 0.30
