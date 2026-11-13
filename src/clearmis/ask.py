"""Answer questions about the sales data without ever executing generated code.

A question becomes a small JSON *plan* (metric, group by, filters, months, top N). The plan
is validated against the real columns and values, then run by fixed pandas code. A question
that cannot be mapped is refused, never guessed. With an LLM configured, it writes the plan;
without one, a keyword planner handles common questions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from .ingest import DIMENSIONS
from .llm import extract_json
from .narrative import inr, pct

Metric = Literal["net_revenue", "orders", "units", "aov", "return_rate", "gross_margin"]
LABEL = {
    "net_revenue": "net revenue",
    "orders": "orders",
    "units": "units",
    "aov": "average order value",
    "return_rate": "return rate",
    "gross_margin": "gross margin",
}


class Plan(BaseModel):
    metric: Metric = "net_revenue"
    group_by: list[str] = Field(default_factory=list, max_length=2)
    filters: dict[str, list[str]] = Field(default_factory=dict)
    months: list[str] = Field(default_factory=list)  # "2026-05"
    top: int | None = Field(default=None, ge=1, le=50)
    ascending: bool = False  # True for "worst", "lowest", "least"

    @field_validator("group_by")
    @classmethod
    def _dims(cls, v):
        bad = [x for x in v if x not in DIMENSIONS and x != "month"]
        if bad:
            raise ValueError(f"unknown dimension {bad}")
        return v

    @field_validator("filters")
    @classmethod
    def _fdims(cls, v):
        bad = [x for x in v if x not in DIMENSIONS]
        if bad:
            raise ValueError(f"unknown filter {bad}")
        return v


class Refused(Exception):
    pass


def resolve(plan: Plan, d: pd.DataFrame) -> Plan:
    """Match filter values and months to what really exists (case-insensitive)."""
    filters = {}
    for dim, vals in plan.filters.items():
        known = {str(x).lower(): str(x) for x in d[dim].unique()}
        hit = [known[v.lower()] for v in vals if v.lower() in known]
        if not hit:
            raise Refused(f"no {dim} called {', '.join(vals)} in the data")
        filters[dim] = hit
    have = {str(m) for m in d["month"].unique()}
    months = [m for m in plan.months if m in have]
    if plan.months and not months:
        raise Refused(f"no data for {', '.join(plan.months)}")
    return plan.model_copy(update={"filters": filters, "months": months})


def metric_value(g: pd.DataFrame, metric: str) -> float:
    sales = g[g.status == "sale"]
    gross = sales["net_amount"].sum()
    orders = sales["order_id"].nunique()
    net = g["net_amount"].sum()
    if metric == "net_revenue":
        return float(net)
    if metric == "orders":
        return float(orders)
    if metric == "units":
        return float(g["qty"].sum())
    if metric == "aov":
        return float(gross / orders) if orders else float("nan")
    if metric == "return_rate":
        return (
            float(-g.loc[g.status == "return", "net_amount"].sum() / gross * 100)
            if gross
            else float("nan")
        )
    return float((net - g["cost"].sum()) / net * 100) if net else float("nan")  # gross margin


def run(plan: Plan, d: pd.DataFrame) -> pd.Series | float:
    if plan.metric == "gross_margin" and d["cost"].isna().all():
        raise Refused("the data has no cost column, so margin cannot be computed")
    x = d
    for dim, vals in plan.filters.items():
        x = x[x[dim].isin(vals)]
    if plan.months:
        x = x[x["month"].astype(str).isin(plan.months)]
    if x.empty:
        raise Refused("no rows match that question")
    if not plan.group_by:
        return metric_value(x, plan.metric)
    keys = [x["month"].astype(str) if g == "month" else x[g] for g in plan.group_by]
    s = x.groupby(keys).apply(lambda g: metric_value(g, plan.metric), include_groups=False)
    s = s.sort_values(ascending=plan.ascending)
    return s.head(plan.top) if plan.top else s


def fmt(metric: str, v: float) -> str:
    if metric in ("net_revenue", "aov"):
        return inr(v)
    if metric in ("return_rate", "gross_margin"):
        return pct(v, False)
    return f"{v:,.0f}"


def answer_text(plan: Plan, res) -> str:
    scope = " ".join(f"{', '.join(v)}" for v in plan.filters.values())
    when = f" in {', '.join(plan.months)}" if plan.months else ""
    head = f"{LABEL[plan.metric].capitalize()}{' for ' + scope if scope else ''}{when}"
    if isinstance(res, float):
        return f"{head}: {fmt(plan.metric, res)}."
    rows = "; ".join(
        f"{k if not isinstance(k, tuple) else ' / '.join(k)}: {fmt(plan.metric, v)}"
        for k, v in res.items()
    )
    return f"{head} by {' and '.join(plan.group_by)}: {rows}."


# -- planners --------------------------------------------------------------------------
MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
METRIC_WORDS = [
    ("aov", r"average order|aov|basket|order value"),
    ("return_rate", r"return rate|returns? (?:%|percent|ratio)|refund rate|returns"),
    ("gross_margin", r"margin|profitab"),
    ("orders", r"\borders?\b|how many orders"),
    ("units", r"\bunits?\b|quantity|pieces|volume"),
    ("net_revenue", r"revenue|sales|turnover|income|earn|sold|business"),
]
DIM_WORDS = {
    "region": r"region|state|zone|area|geograph",
    "channel": r"channel|platform|"
    r"marketplace|source",
    "category": r"categor|segment|product line",
    "sku": r"\bsku",
    "product": r"product|item",
    "month": r"month|monthly|trend",
}


STOP = {
    "by",
    "for",
    "in",
    "of",
    "last",
    "this",
    "with",
    "and",
    "the",
    "wise",
    "level",
    "is",
    "was",
    "had",
    "has",
    "trend",
    "breakdown",
    "split",
    "mix",
    "performance",
    "sales",
    "revenue",
    "orders",
    "share",
}
DIM_ALIASES = {
    "region",
    "regions",
    "state",
    "states",
    "channel",
    "channels",
    "category",
    "categories",
    "product",
    "products",
    "platform",
    "platforms",
}


def keyword_plan(q: str, d: pd.DataFrame) -> Plan | None:
    s = q.lower()
    metric = next((m for m, pat in METRIC_WORDS if re.search(pat, s)), None)
    if metric is None:
        return None
    group = []
    for m in re.finditer(r"\b(?:by|per|each|across|split by|for every|which)\s+(\w+)", s):
        for dim, pat in DIM_WORDS.items():
            if re.match(pat, m.group(1)) and dim not in group:
                group.append(dim)
    if re.search(r"\bmonthly\b|\btrend\b|over time", s) and "month" not in group:
        group.append("month")
    if re.search(r"\b(best|top|worst|highest|lowest|most|least)\b", s) and not group:
        for dim, pat in DIM_WORDS.items():
            if re.search(pat, s):
                group.append(dim)
                break
    filters: dict[str, list[str]] = {}
    for dim in DIMENSIONS:
        for v in d[dim].unique():
            v = str(v)
            if len(v) > 2 and v != "Unknown" and re.search(rf"\b{re.escape(v.lower())}\b", s):
                filters.setdefault(dim, []).append(v)
    # "region Atlantis": a named (capitalised) value that does not exist must refuse
    for m in re.finditer(
        r"(?i:\b(region|state|channel|platform|category|product))\s+([A-Z][\w&'-]*)", q
    ):
        word = m.group(2)
        dim = {"state": "region", "platform": "channel"}.get(m.group(1).lower(), m.group(1).lower())
        if word.lower() in STOP or word.lower() in DIM_ALIASES:
            continue
        if not any(str(v).lower().startswith(word.lower()) for v in d[dim].unique()):
            filters[dim] = [word]
    for dim in list(filters):  # a mention that names the grouping dimension is not a filter
        if dim in group and len(filters[dim]) > 1:
            del filters[dim]
    have = sorted(str(m) for m in d["month"].unique())
    months = []
    if re.search(r"last month|previous month|latest month", s):
        months = [have[-1]]
    for m in re.finditer(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*"
        r"(?:'|20)?(\d{2})?\b",
        s,
    ):
        mon = MONTHS[m.group(1)]
        yr = m.group(2)
        cands = [h for h in have if int(h[5:]) == mon and (not yr or h[2:4] == yr)]
        if cands:
            months.append(cands[-1])
    top = re.search(r"\btop\s+(\d+)", s)
    n = (
        int(top.group(1))
        if top
        else (
            1
            if re.search(
                r"\b(best|worst|highest|lowest|most|"
                r"least|which)\b",
                s,
            )
            and group
            else None
        )
    )
    if n and re.search(r"\b(worst|lowest|least)\b", s):
        n = -n
    return Plan(
        metric=metric,
        group_by=group,
        filters=filters,
        months=months,
        top=abs(n) if n else None,
        ascending=bool(n and n < 0),
    )


PLAN_PROMPT = """Turn the question into a JSON query plan for a sales table. Reply with JSON only.
Schema: {{"metric": one of {metrics}, "group_by": list of up to 2 of {dims}, "filters":
{{dimension: [values]}}, "months": ["YYYY-MM", ...], "top": integer or null}}.
Known values: {values}
Months available: {months}
If the question cannot be answered from this table, reply {{"refuse": "reason"}}.
Question: {q}"""


def llm_plan(q: str, d: pd.DataFrame, ask: Callable[[str, float], str | None]) -> Plan | None:
    values = {dim: sorted(map(str, d[dim].unique()))[:40] for dim in DIMENSIONS}
    months = sorted(str(m) for m in d["month"].unique())
    raw = extract_json(
        ask(
            PLAN_PROMPT.format(
                metrics=list(LABEL), dims=DIMENSIONS + ["month"], values=values, months=months, q=q
            ),
            0.0,
        )
    )
    if not raw or "refuse" in raw:
        return None
    try:
        return Plan(**raw)
    except ValidationError:
        return None


def ask_question(q: str, d: pd.DataFrame, ask: Callable | None = None) -> dict:
    plan = llm_plan(q, d, ask) if ask else keyword_plan(q, d)
    if plan is None:
        return {
            "ok": False,
            "answer": "I couldn't map that question to this data. Try naming a "
            "metric (revenue, orders, units, AOV, return rate, margin) and a breakdown.",
            "plan": None,
        }
    try:
        plan = resolve(plan, d)
        res = run(plan, d)
    except Refused as e:
        return {"ok": False, "answer": f"Can't answer: {e}.", "plan": plan.model_dump()}
    value = (
        res
        if isinstance(res, float)
        else {(" / ".join(k) if isinstance(k, tuple) else str(k)): float(v) for k, v in res.items()}
    )
    return {"ok": True, "answer": answer_text(plan, res), "plan": plan.model_dump(), "value": value}
