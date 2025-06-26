import json

import pandas as pd

from nanotron_data.quality import (
    LineageEmitter,
    OpenLineageEvent,
    Suite,
    expect_column_max_le,
    expect_column_min_ge,
    expect_column_no_null,
    expect_column_unique,
    expect_increasing_index,
    expect_returns_in_range,
    expect_row_count_between,
)


def test_no_null_passes_on_clean_frame():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert expect_column_no_null("x")(df).passed


def test_no_null_fails_with_one_nan():
    df = pd.DataFrame({"x": [1, None, 3]})
    res = expect_column_no_null("x")(df)
    assert not res.passed
    assert res.metric == 1


def test_returns_in_range_flags_outliers():
    df = pd.DataFrame({"ret": [0.01, -0.02, 0.7, -0.6]})
    res = expect_returns_in_range()(df)
    assert not res.passed
    assert res.metric == 2


def test_increasing_index_passes():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
    assert expect_increasing_index()(df).passed


def test_suite_runs_all_checks_and_summarizes():
    df = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5], "ret": [0.01, -0.02, 0.001, 0.0, 0.005]}
    )
    suite = (
        Suite("bars")
        .add(expect_column_no_null("x"))
        .add(expect_column_min_ge("x", 0))
        .add(expect_column_max_le("x", 100))
        .add(expect_column_unique("x"))
        .add(expect_row_count_between(1, 10))
        .add(expect_returns_in_range("ret"))
    )
    results = suite.run(df)
    assert all(r.passed for r in results)
    assert suite.all_passed(df)


def test_lineage_emitter_writes_ndjson(tmp_path):
    log = tmp_path / "ol.ndjson"
    em = LineageEmitter(log_path=log)
    rid = em.emit_complete(
        job_namespace="research",
        job_name="bars_to_features",
        inputs=["bars/AAPL"],
        outputs=["features/momo_5d"],
        facets={"git_sha": "abc123"},
    )
    line = log.read_text().strip()
    payload = json.loads(line)
    assert payload["job"]["name"] == "bars_to_features"
    assert payload["run"]["runId"] == rid
    assert payload["inputs"][0]["name"] == "bars/AAPL"
    assert payload["job"]["facets"]["git_sha"] == "abc123"
