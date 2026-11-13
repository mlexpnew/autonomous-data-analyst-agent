"""Read a sales export (Shopify, Tally, marketplace or any sheet) into one canonical table.

Handles Indian formats: ₹ / Rs. / INR prefixes, lakh grouping ("1,23,456.50"), brackets for
negatives and day-first dates ("05/03/2026" is 5 March).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = [
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
DIMENSIONS = ["region", "channel", "category", "product", "sku"]

ALIASES: dict[str, list[str]] = {
    "date": [
        "date",
        "order date",
        "invoice date",
        "created at",
        "voucher date",
        "order_date",
        "transaction date",
        "bill date",
    ],
    "order_id": [
        "order id",
        "order_id",
        "order no",
        "order number",
        "name",
        "voucher no",
        "voucher no.",
        "invoice no",
        "invoice number",
        "bill no",
    ],
    "customer": [
        "customer",
        "email",
        "customer email",
        "particulars",
        "party",
        "party name",
        "buyer",
        "customer name",
    ],
    "sku": ["sku", "lineitem sku", "item code", "product code", "asin"],
    "product": [
        "product",
        "product name",
        "lineitem name",
        "item",
        "stock item",
        "item name",
        "description",
    ],
    "category": ["category", "product type", "product category", "group", "stock group"],
    "region": [
        "region",
        "state",
        "shipping province",
        "shipping province name",
        "zone",
        "ship state",
        "city",
    ],
    "channel": ["channel", "source", "platform", "marketplace", "sales channel"],
    "qty": ["qty", "quantity", "lineitem quantity", "units", "billed qty"],
    "unit_price": ["unit price", "price", "rate", "lineitem price", "selling price", "unit_price"],
    "discount": ["discount", "lineitem discount", "discount amount"],
    "net_amount": [
        "net amount",
        "net_amount",
        "amount",
        "value",
        "net sales",
        "revenue",
        "total",
        "line total",
        "taxable value",
        "item total",
    ],
    "cost": ["cost", "cogs", "cost price", "purchase cost", "landed cost"],
    "status": ["status", "financial status", "order status", "voucher type", "type"],
    "currency": ["currency", "currency code"],
}


def _key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


def map_columns(columns: list[str]) -> dict[str, str]:
    """source column -> canonical name (first alias match wins, each canonical used once)."""
    found: dict[str, str] = {}
    keys = {c: _key(c) for c in columns}
    for canon, aliases in ALIASES.items():
        al = [_key(a) for a in aliases]
        for a in al:
            hit = next((c for c, k in keys.items() if k == a and c not in found), None)
            if hit:
                found[hit] = canon
                break
    # Shopify puts the *order* total on the first line item only: use line prices instead
    line_price = any(keys[c].startswith("lineitem") and v == "unit_price" for c, v in found.items())
    if line_price:
        found = {c: v for c, v in found.items() if not (v == "net_amount" and keys[c] == "total")}
    return found


_NUM = re.compile(r"[^\d.\-]")


def parse_number(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, int | float | np.number):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "-", "null"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"(?i)\b(?:rs\.?|inr|usd)|₹|\$", "", s)
    s = _NUM.sub("", s)
    if s in ("", "-", ".", "-."):
        return np.nan
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -abs(x) if neg else x


ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def parse_dates(col: pd.Series) -> pd.Series:
    """ISO dates as written; everything else day-first (05/03/2026 is 5 March, Indian style)."""
    s = col.astype(str).str.strip()
    iso = s.str.match(ISO)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], format="ISO8601", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], format="mixed", dayfirst=True, errors="coerce")
    return out


STATUS = [
    ("test", r"\btest\b"),
    ("cancelled", r"cancel|void|reject"),
    ("return", r"return|refund|credit note|rto"),
    ("sale", r"paid|complete|fulfil|deliver|sale|ship|success|invoice|pending|authori"),
]


def normalise_status(v) -> str:
    s = str(v or "").lower()
    for name, pat in STATUS:
        if re.search(pat, s):
            return name
    return "sale"


def load(path_or_df, base_currency: str = "INR") -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (canonical DataFrame, column mapping used)."""
    if isinstance(path_or_df, pd.DataFrame):
        raw = path_or_df.copy()
    else:
        p = Path(path_or_df)
        raw = (
            pd.read_excel(p)
            if p.suffix.lower() in (".xlsx", ".xls")
            else pd.read_csv(p, dtype=str, keep_default_na=False)
        )
    raw = raw.astype(object)  # same behaviour on pandas 2 and pandas 3 (pyarrow strings)
    mapping = map_columns(list(raw.columns))
    df = raw.rename(columns=mapping)[list(mapping.values())].copy()
    for c in CANONICAL:
        if c not in df.columns:
            df[c] = np.nan
    df = df[CANONICAL]
    for c in ("qty", "unit_price", "discount", "net_amount", "cost"):
        df[c] = df[c].astype(object).map(parse_number).astype(float)
    df["date"] = parse_dates(df["date"])
    if df["net_amount"].isna().all():  # e.g. Shopify line items: compute from price
        df["net_amount"] = df["qty"] * df["unit_price"] - df["discount"].fillna(0)
    df["status"] = df["status"].map(normalise_status)
    df["currency"] = (
        df["currency"].replace("", np.nan).fillna(base_currency).astype(str).str.upper().str.strip()
    )
    for c in ("order_id", "customer", "sku", "product", "category", "region", "channel"):
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["qty"] = df["qty"].fillna(1)
    return df, mapping
