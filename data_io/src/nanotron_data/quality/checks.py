"""Lightweight data-quality checks.

A self-contained Great-Expectations-style suite — same idiom (a Check
returns a CheckResult; suites compose) but a fraction of the
dependency footprint.  Easy to swap out for the real ``great_expectations``
when the team needs the full Data Docs + Datasource catalog story.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    metric: float | int | None = None
    detail: str = ""


@dataclass
class Check:
    name: str
    fn: Callable[[pd.DataFrame], CheckResult]

    def __call__(self, df: pd.DataFrame) -> CheckResult:
        return self.fn(df)


@dataclass
class Suite:
    name: str
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> "Suite":
        self.checks.append(check)
        return self

    def run(self, df: pd.DataFrame) -> list[CheckResult]:
        return [c(df) for c in self.checks]

    def all_passed(self, df: pd.DataFrame) -> bool:
        return all(r.passed for r in self.run(df))


# ---- builders that produce Checks with descriptive names ------------------

def expect_column_no_null(column: str) -> Check:
    def _fn(df: pd.DataFrame) -> CheckResult:
        n = int(df[column].isna().sum())
        return CheckResult(
            name=f"no_null:{column}",
            passed=n == 0,
            metric=n,
            detail=f"{n} nulls in {column}" if n else "ok",
        )

    return Check(name=f"no_null:{column}", fn=_fn)


def expect_column_min_ge(column: str, low: float) -> Check:
    def _fn(df: pd.DataFrame) -> CheckResult:
        m = float(df[column].min())
        return CheckResult(
            name=f"min_ge:{column}>={low}",
            passed=m >= low,
            metric=m,
            detail=f"min={m}",
        )

    return Check(name=f"min_ge:{column}", fn=_fn)


def expect_column_max_le(column: str, high: float) -> Check:
    def _fn(df: pd.DataFrame) -> CheckResult:
        m = float(df[column].max())
        return CheckResult(
            name=f"max_le:{column}<={high}",
            passed=m <= high,
            metric=m,
            detail=f"max={m}",
        )

    return Check(name=f"max_le:{column}", fn=_fn)


def expect_column_unique(column: str) -> Check:
    def _fn(df: pd.DataFrame) -> CheckResult:
        dup = int(df[column].duplicated().sum())
        return CheckResult(
            name=f"unique:{column}",
            passed=dup == 0,
            metric=dup,
            detail=f"{dup} duplicates" if dup else "ok",
        )

    return Check(name=f"unique:{column}", fn=_fn)


def expect_increasing_index() -> Check:
    def _fn(df: pd.DataFrame) -> CheckResult:
        ok = bool(df.index.is_monotonic_increasing)
        return CheckResult(
            name="increasing_index",
            passed=ok,
            metric=None,
            detail="ok" if ok else "non-monotone index",
        )

    return Check(name="increasing_index", fn=_fn)


def expect_row_count_between(low: int, high: int) -> Check:
    def _fn(df: pd.DataFrame) -> CheckResult:
        n = len(df)
        return CheckResult(
            name=f"row_count:[{low},{high}]",
            passed=low <= n <= high,
            metric=n,
            detail=f"n={n}",
        )

    return Check(name="row_count", fn=_fn)


def expect_returns_in_range(
    column: str = "ret",
    lower: float = -0.5,
    upper: float = 0.5,
) -> Check:
    """Catch obviously-corrupted returns: any single-period |ret| > 50% is
    almost certainly bad data on equities/FX (crypto users should widen
    the range)."""

    def _fn(df: pd.DataFrame) -> CheckResult:
        out_of_range = int(((df[column] < lower) | (df[column] > upper)).sum())
        return CheckResult(
            name=f"returns_in_range:{column}∈[{lower},{upper}]",
            passed=out_of_range == 0,
            metric=out_of_range,
            detail=f"{out_of_range} out-of-range rows",
        )

    return Check(name=f"returns_in_range:{column}", fn=_fn)
