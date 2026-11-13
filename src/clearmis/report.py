"""From a raw export to the monthly MIS: summary, KPIs, drivers, unusual days, data fixes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import Anomaly, Driver, anomalies, drivers, monthly_kpis
from .ingest import load
from .narrative import Sentence, build, render
from .quality import Issue, clean


def json_safe(x):
    """NaN and infinity become None so any JSON encoder accepts the result."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [json_safe(v) for v in x]
    if isinstance(x, float | np.floating):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


@dataclass
class Report:
    month: pd.Period
    summary: str
    sentences: list[Sentence]
    kpis: pd.DataFrame
    total_change: float
    drivers: list[Driver]
    anomalies: list[Anomaly]
    issues: list[Issue]
    clean: pd.DataFrame
    mapping: dict[str, str]
    dropped_sentences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        k = self.kpis.reset_index()
        k["month"] = k["month"].astype(str)
        return json_safe(
            {
                "month": str(self.month),
                "summary": self.summary,
                "kpis": k.round(2).to_dict(orient="records"),
                "change_vs_previous_month": round(self.total_change, 2),
                "drivers": [x.as_dict() for x in self.drivers],
                "unusual_days": [a.as_dict() for a in self.anomalies],
                "data_fixes": [i.as_dict() for i in self.issues],
                "columns_mapped": self.mapping,
                "dropped_unverified_sentences": self.dropped_sentences,
            }
        )


def make(
    source,
    month: str | None = None,
    as_of: str | None = None,
    fx: dict[str, float] | None = None,
    rewrite: Callable[[str], str | None] | None = None,
) -> Report:
    raw, mapping = load(source)
    stamp = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    d, issues = clean(raw, as_of=stamp, fx=fx)
    if d.empty:
        raise ValueError("no usable rows after cleaning")
    kpis = monthly_kpis(d)
    if month:
        m = pd.Period(month, "M")
    else:  # latest *complete* month: skip the current one if it is still running
        m = kpis.index.max()
        if m == stamp.to_period("M") and len(kpis) > 1:
            m = kpis.index[-2]
    change, drv = drivers(d, m) if (m - 1) in kpis.index else (0.0, [])
    anom = anomalies(d)
    sentences = build(kpis, m, change, drv, anom, issues)
    summary, dropped = render(sentences, rewrite)
    return Report(m, summary, sentences, kpis, change, drv, anom, issues, d, mapping, dropped)


def to_excel(r: Report, path: str | Path) -> Path:
    path = Path(path)
    k = r.kpis.reset_index()
    k["month"] = k["month"].astype(str)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Summary": [s.text for s in r.sentences]}).to_excel(
            xw, sheet_name="Summary", index=False
        )
        k.round(2).to_excel(xw, sheet_name="Monthly KPIs", index=False)
        pd.DataFrame([x.as_dict() for x in r.drivers]).to_excel(
            xw, sheet_name="Drivers", index=False
        )
        pd.DataFrame([a.as_dict() for a in r.anomalies]).to_excel(
            xw, sheet_name="Unusual days", index=False
        )
        pd.DataFrame([i.as_dict() for i in r.issues]).to_excel(
            xw, sheet_name="Data fixes", index=False
        )
        c = r.clean.copy()
        c["month"] = c["month"].astype(str)
        c.to_excel(xw, sheet_name="Clean ledger", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    60, max(10, max(len(str(c.value or "")) for c in col[:200]) + 2)
                )
    return path
