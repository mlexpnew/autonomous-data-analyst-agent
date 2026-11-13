import numpy as np
import pandas as pd

from clearmis.analysis import anomalies, drivers, monthly_kpis
from clearmis.narrative import Sentence, inr, numbers, render, verify


def _ledger(event=True, spike_day=None):
    rng = np.random.default_rng(0)
    rows = []
    for day in pd.date_range("2026-01-01", "2026-04-30"):
        n = 40
        if spike_day and day == pd.Timestamp(spike_day):
            n = 120
        for k in range(n):
            region = ["North", "South", "East", "West"][k % 4]
            amt = 1000 * rng.uniform(0.9, 1.1)
            if event and day.month == 4 and region == "East":
                amt *= 1.6
            rows.append(
                [
                    day,
                    f"{day:%m%d}-{k}",
                    f"c{k}",
                    "S",
                    "P",
                    "Cat",
                    region,
                    "Web",
                    1,
                    amt,
                    0,
                    amt,
                    amt * 0.5,
                    "sale",
                    "INR",
                ]
            )
    cols = [
        "date",
        "order_id",
        "customer",
        "sku",
        "product",
        "category",
        "region",
        "channel",
        "qty",
        "unit_price",
        "discount",
        "net_amount",
        "cost",
        "status",
        "currency",
    ]
    d = pd.DataFrame(rows, columns=cols)
    d["month"] = d["date"].dt.to_period("M")
    return d


def test_driver_beyond_trend():
    d = _ledger()
    change, top = drivers(d, pd.Period("2026-04", "M"))
    assert (top[0].dimension, top[0].member) == ("region", "East") and change > 0


def test_unusual_day_found_and_quiet_days_not():
    d = _ledger(event=False, spike_day="2026-03-18")
    found = [a.date for a in anomalies(d)]
    assert found == ["2026-03-18"]


def test_kpis():
    k = monthly_kpis(_ledger(event=False))
    assert list(k.index.astype(str)) == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert abs(k.loc[pd.Period("2026-02", "M"), "gross_margin_pct"] - 50) < 1e-6


def test_inr_and_number_kinds():
    assert (
        inr(4_25_00_000) == "₹4.25 crore"
        and inr(12_43_210) == "₹12.4 lakh"
        and inr(86500) == "₹86,500"
    )
    kinds = [k for *_, k in numbers("In May 2026 SK-07 made ₹12.4 lakh, +3.1% and 1,234 orders")]
    assert kinds == ["year", "money", "pct", "count"]


def test_verify_is_type_aware_and_rounding_aware():
    facts = [(1243210.0, "money"), (3.1, "pct"), (1234.0, "count"), (2026.0, "year")]
    assert verify("May 2026: ₹12.4 lakh, +3.1%, 1,234 orders", facts) == []
    assert verify("₹12.9 lakh", facts) == ["₹12.9 lakh"]
    assert verify("1234%", facts) == ["1234%"]  # right number, wrong kind
    assert verify("May 2027", facts) == ["2027"]


def test_llm_rewrite_with_invented_number_is_dropped():
    s = [Sentence("Revenue was ₹12.4 lakh.", [(1243210.0, "money")])]
    text, dropped = render(s, lambda t: "Revenue hit ₹12.4 lakh. Next month should reach ₹15 lakh.")
    assert text == "Revenue hit ₹12.4 lakh." and dropped == ["Next month should reach ₹15 lakh."]
