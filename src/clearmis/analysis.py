"""Monthly KPIs, what drove the change, and unusual days."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HIERARCHY = [("category", "product")]


def monthly_kpis(d: pd.DataFrame) -> pd.DataFrame:
    sales, rets = d[d.status == "sale"], d[d.status == "return"]
    g = pd.DataFrame(
        {
            "net_revenue": d.groupby("month")["net_amount"].sum(),
            "gross_sales": sales.groupby("month")["net_amount"].sum(),
            "returns": rets.groupby("month")["net_amount"].sum(),
            "orders": sales.groupby("month")["order_id"].nunique(),
            "units": d.groupby("month")["qty"].sum(),
            "customers": sales.groupby("month")["customer"].nunique(),
        }
    ).fillna(0)
    g["aov"] = g["gross_sales"] / g["orders"].replace(0, np.nan)
    g["return_rate_pct"] = -g["returns"] / g["gross_sales"].replace(0, np.nan) * 100
    if d["cost"].notna().any():
        cost = d.groupby("month")["cost"].sum(min_count=1)
        g["gross_margin_pct"] = (g["net_revenue"] - cost) / g["net_revenue"] * 100
    g["mom_pct"] = g["net_revenue"].pct_change() * 100
    g["yoy_pct"] = g["net_revenue"].pct_change(12) * 100 if len(g) > 12 else np.nan
    return g


@dataclass
class Driver:
    dimension: str
    member: str
    previous: float
    current: float
    change: float
    beyond_trend: float  # change not explained by how the rest of the business moved
    share_of_total_change: float

    def as_dict(self) -> dict:
        return asdict(self)


def drivers(
    d: pd.DataFrame, month: pd.Period, dims: list[str] | None = None, top: int = 5
) -> tuple[float, list[Driver]]:
    """Rank members by how much their change differs from the rest of the business."""
    dims = dims or ["region", "channel", "category", "product"]
    prev_m = month - 1
    cur = d[d.month == month]
    prv = d[d.month == prev_m]
    tot_c, tot_p = cur["net_amount"].sum(), prv["net_amount"].sum()
    total_change = tot_c - tot_p
    out: list[Driver] = []
    for dim in dims:
        if d[dim].nunique() < 2:
            continue
        c = cur.groupby(dim)["net_amount"].sum()
        p = prv.groupby(dim)["net_amount"].sum()
        idx = c.index.union(p.index)
        c, p = c.reindex(idx, fill_value=0), p.reindex(idx, fill_value=0)
        rest_c, rest_p = tot_c - c, tot_p - p
        growth = np.where(rest_p > 0, rest_c / rest_p.replace(0, np.nan), 1.0)
        beyond = c - p * np.nan_to_num(growth, nan=1.0)
        for m in idx:
            out.append(
                Driver(
                    dim,
                    str(m),
                    float(p[m]),
                    float(c[m]),
                    float(c[m] - p[m]),
                    float(beyond[m]),
                    float((c[m] - p[m]) / total_change) if total_change else 0.0,
                )
            )
    out.sort(key=lambda x: -abs(x.beyond_trend))
    # Prefer the most specific explanation: a SKU that carries its category's movement
    if out:
        best = out[0]
        for parent, child in HIERARCHY:
            if best.dimension != parent:
                continue
            members = set(cur.loc[cur[parent] == best.member, child]) | set(
                prv.loc[prv[parent] == best.member, child]
            )
            kids = [x for x in out if x.dimension == child and x.member in members]
            if kids and abs(kids[0].beyond_trend) >= 0.8 * abs(best.beyond_trend):
                out.remove(kids[0])
                out.insert(0, kids[0])
                best = kids[0]
    return float(total_change), out[:top]


@dataclass
class Anomaly:
    date: str
    value: float
    expected: float
    change_pct: float
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


def anomalies(
    d: pd.DataFrame, z: float = 4.0, min_change_pct: float = 30.0, weeks: int = 8
) -> list[Anomaly]:
    """Days far from normal for that weekday at the current level of the business.

    Expected = median of the same weekday over the last 8 weeks, scaled by how the last 7
    days compare with those 8 weeks (so a festive build-up or a slow month is not "unusual").
    Normal variation = MAD of all 56 days after removing the weekday pattern: a far steadier
    estimate than 8 same-weekday points.
    """
    daily = d.groupby(d["date"].dt.normalize())["net_amount"].sum()
    if daily.empty:
        return []
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    out = []
    vals = daily.to_numpy()
    span = 7 * weeks
    for i in range(7 * 4, len(vals)):
        n = min(span, i - i % 7 if i < span else span)
        window = vals[i - n : i]
        wk = np.array([np.median(window[g::7]) for g in range(7)])  # g=0: same weekday as day i
        if (wk <= 0).any():
            continue
        rel = window / wk[np.arange(n) % 7] - 1
        sigma = max(1.4826 * float(np.median(np.abs(rel - np.median(rel)))), 0.03)
        base = float(np.median(window))
        level = float(np.median(vals[i - 7 : i])) / base if base > 0 else 1.0
        expected = float(wk[0]) * level
        if expected <= 0:
            continue
        score = (vals[i] / expected - 1) / sigma
        pct = (vals[i] / expected - 1) * 100
        if abs(score) >= z and abs(pct) >= min_change_pct:
            out.append(
                Anomaly(str(daily.index[i].date()), float(vals[i]), expected, pct, float(score))
            )
    return out
