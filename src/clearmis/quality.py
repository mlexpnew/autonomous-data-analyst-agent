"""Data quality checks and cleaning. Every fix is counted, valued and explained.

Raw exports overstate sales: repeated rows, test and cancelled orders, returns recorded as
positive amounts, other currencies mixed in. This module turns the raw table into the ledger
the MIS is computed from, and reports exactly what it changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Issue:
    code: str
    title: str
    rows: int
    amount: float  # net amount affected, in base currency where known
    action: str

    def as_dict(self) -> dict:
        return asdict(self)


def clean(
    df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    base_currency: str = "INR",
    fx: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[Issue]]:
    d = df.copy()
    issues: list[Issue] = []
    fx = {k.upper(): v for k, v in (fx or {}).items()}

    def take(mask, code, title, action, drop=True):
        nonlocal d
        n = int(mask.sum())
        if n:
            issues.append(
                Issue(code, title, n, round(float(d.loc[mask, "net_amount"].sum()), 2), action)
            )
            if drop:
                d = d[~mask]

    take(
        d.duplicated(keep="first"),
        "duplicate_rows",
        "Exact duplicate rows",
        "removed (kept the first copy)",
    )
    take(d["date"].isna(), "bad_date", "Rows without a readable date", "removed")
    take(d["net_amount"].isna(), "missing_amount", "Rows without an amount or price", "removed")
    is_test = (
        (d["status"] == "test")
        | d["customer"].str.contains(r"\btest\b", case=False)
        | (d["order_id"].str.contains(r"\btest\b", case=False))
    )
    take(is_test, "test_orders", "Test orders", "excluded")
    take(d["status"] == "cancelled", "cancelled", "Cancelled orders", "excluded")
    if as_of is not None:
        take(d["date"] > as_of, "future_date", "Rows dated in the future", "excluded")
    # stray dates far outside the period the data really covers (typos, wrong year)
    if len(d):
        per = d["date"].dt.to_period("M")
        counts = per.value_counts().sort_index()
        dense = counts[counts >= 0.05 * counts.median()].index
        if len(dense):
            lo, hi = dense.min(), dense.max()
            take(
                (per < lo) | (per > hi),
                "outlying_date",
                "Rows dated outside the period the data covers",
                "excluded",
            )
    other = d["currency"] != base_currency.upper()
    if other.any():
        convertible = other & d["currency"].isin(fx)
        if convertible.any():
            rate = d.loc[convertible, "currency"].astype(object).map(fx).astype(float)
            for col in ("net_amount", "unit_price", "discount", "cost"):
                d.loc[convertible, col] = d.loc[convertible, col].astype(float) * rate
            issues.append(
                Issue(
                    "converted_currency",
                    "Rows in another currency",
                    int(convertible.sum()),
                    round(float(d.loc[convertible, "net_amount"].sum()), 2),
                    f"converted to {base_currency} at the given rates",
                )
            )
            d.loc[convertible, "currency"] = base_currency.upper()
        take(
            d["currency"] != base_currency.upper(),
            "other_currency",
            f"Rows not in {base_currency} and no exchange rate given",
            "excluded",
        )

    ret = d["status"] == "return"
    pos = ret & (d["net_amount"] > 0)
    take(pos, "return_sign", "Returns recorded as positive amounts", "turned negative", drop=False)
    for col in ("net_amount", "qty", "cost"):
        d.loc[ret, col] = -d.loc[ret, col].abs()
    neg_sale = (d["status"] == "sale") & (d["net_amount"] < 0)
    take(
        neg_sale, "negative_sale", "Negative amounts on sale rows", "treated as returns", drop=False
    )
    d.loc[neg_sale, "status"] = "return"
    for dim in ("region", "channel", "category", "product", "sku"):
        blank = d[dim].isin(["", "nan"])
        if blank.any() and not blank.all():
            take(blank, f"missing_{dim}", f"Rows without a {dim}", "kept as 'Unknown'", drop=False)
        d.loc[blank, dim] = "Unknown"
    d["month"] = d["date"].dt.to_period("M")
    return d.reset_index(drop=True), issues
