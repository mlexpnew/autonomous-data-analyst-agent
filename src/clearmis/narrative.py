"""Executive summary where every number is checked against the computed results.

Sentences are built from facts. An optional LLM may rephrase them, but its text is then
verified: any sentence carrying a number that is not one of the facts is dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


def inr(x: float) -> str:
    """₹ in Indian units: ₹4.25 crore, ₹12.4 lakh, ₹86,500."""
    a, sign = abs(x), "-" if x < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.1f} lakh"
    return f"{sign}₹{a:,.0f}"  # below a lakh, Indian and western grouping are the same


def pct(x: float, signed: bool = True) -> str:
    x = 0.0 if abs(x) < 0.05 else x  # never print "-0.0%"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


@dataclass
class Sentence:
    text: str
    facts: list[tuple[float, str]] = field(default_factory=list)  # (value, kind)


def M(x) -> tuple[float, str]:
    return (float(x), "money")


def P(x) -> tuple[float, str]:
    return (float(x), "pct")


def C(x) -> tuple[float, str]:
    return (float(x), "count")


def build(
    kpis: pd.DataFrame,
    month: pd.Period,
    total_change: float,
    drivers: list,
    anomalies: list,
    issues: list,
) -> list[Sentence]:
    k = kpis.loc[month]
    name = month.strftime("%B %Y")
    s: list[Sentence] = []
    rev = float(k.net_revenue)
    line = f"Net revenue for {name} was {inr(rev)}"
    facts = [M(rev), (float(month.year), "year")]
    if pd.notna(k.mom_pct):
        line += f", {pct(k.mom_pct)} on the previous month"
        facts.append(P(k.mom_pct))
    if "yoy_pct" in k and pd.notna(k.yoy_pct):
        line += f" and {pct(k.yoy_pct)} on the same month last year"
        facts.append(P(k.yoy_pct))
    s.append(Sentence(line + ".", facts))
    s.append(
        Sentence(
            f"{int(k.orders):,} orders at an average of {inr(float(k.aov))}; returns "
            f"took back {inr(-float(k.returns))} ({pct(float(k.return_rate_pct), False)}"
            " of gross sales).",
            [C(k.orders), M(k.aov), M(-k.returns), P(k.return_rate_pct)],
        )
    )
    if "gross_margin_pct" in k and pd.notna(k.gross_margin_pct):
        s.append(
            Sentence(
                f"Gross margin was {pct(float(k.gross_margin_pct), False)}.",
                [P(k.gross_margin_pct)],
            )
        )
    if drivers:
        b = drivers[0]
        moved = "rose" if b.change > 0 else "fell"
        s.append(
            Sentence(
                f"The biggest driver was {b.dimension} {b.member}, which {moved} from "
                f"{inr(b.previous)} to {inr(b.current)}, {inr(abs(b.beyond_trend))} "
                f"{'above' if b.beyond_trend > 0 else 'below'} the trend of the rest of the "
                "business.",
                [M(b.previous), M(b.current), M(abs(b.beyond_trend))],
            )
        )
    in_month = [a for a in anomalies if pd.Period(a.date, "M") == month]
    for a in in_month[:2]:
        day = pd.Timestamp(a.date).strftime("%d %B").lstrip("0")
        s.append(
            Sentence(
                f"Unusual day: {day} brought in {inr(a.value)} against a normal "
                f"{inr(a.expected)} for that weekday ({pct(a.change_pct)}).",
                [M(a.value), M(a.expected), P(a.change_pct), C(pd.Timestamp(a.date).day)],
            )
        )
    big = sorted(issues, key=lambda i: -abs(i.amount))[:2]
    if big:
        txt = "; ".join(f"{i.title.lower()}: {i.rows:,} rows, {inr(abs(i.amount))}" for i in big)
        s.append(
            Sentence(
                f"Data fixed before reporting ({txt}).",
                [v for i in big for v in (C(i.rows), M(abs(i.amount)))],
            )
        )
    return s


# -- verification ----------------------------------------------------------------------
NUM = re.compile(
    r"(?<![A-Za-z0-9_.\-/])(?P<cur>₹|rs\.?\s?|inr\s?)?(?P<num>[+-]?\d[\d,]*(?:\.\d+)?)\s?"
    r"(?P<unit>crore|cr\b|lakh|lac\b|l\b|k\b|mn\b|million|%)?(?(unit)|(?![A-Za-z]))",
    re.I,
)
SCALE = {
    "crore": 1e7,
    "cr": 1e7,
    "lakh": 1e5,
    "lac": 1e5,
    "l": 1e5,
    "k": 1e3,
    "mn": 1e6,
    "million": 1e6,
}
MONTH = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+$",
    re.I,
)


def numbers(text: str) -> list[tuple[float, float, str, str]]:
    """(value, tolerance from the shown precision, raw text, kind) for every number."""
    out = []
    for m in NUM.finditer(text):
        raw = m.group("num")
        unit = (m.group("unit") or "").lower()
        v = float(raw.replace(",", ""))
        dec = len(raw.split(".")[1]) if "." in raw else 0
        scale = SCALE.get(unit, 1.0)
        tol = 0.5 * 10**-dec * scale * 1.02 + 1e-9
        if unit == "%":
            kind = "pct"
        elif m.group("cur") or unit in SCALE:
            kind = "money"
        elif 1900 <= v <= 2100 and dec == 0 and MONTH.search(text[: m.start()]):
            kind = "year"
        else:
            kind = "count"
        out.append((v * scale, tol, m.group(0).strip(), kind))
    return out


def verify(text: str, facts: list) -> list[str]:
    """Numbers in text that match no fact of the same kind (within shown rounding). [] = clean."""
    typed = [f if isinstance(f, tuple) else (float(f), None) for f in facts]
    bad = []
    for v, tol, raw, kind in numbers(text):
        ok = any(abs(abs(v) - abs(f)) <= tol and (k is None or k == kind) for f, k in typed)
        if not ok:
            bad.append(raw)
    return bad


def render(
    sentences: list[Sentence], rewrite: Callable[[str], str | None] | None = None
) -> tuple[str, list[str]]:
    """Final summary text and the list of dropped (unverifiable) sentences."""
    facts = [f for s in sentences for f in s.facts]
    base = " ".join(s.text for s in sentences)
    if rewrite is None:
        return base, []
    draft = rewrite(base)
    if not draft:
        return base, []
    kept, dropped = [], []
    for sent in re.split(r"(?<=[.!?])\s+", draft.strip()):
        (dropped if verify(sent, facts) else kept).append(sent)
    return (" ".join(kept) if kept else base), dropped
