"""Synthetic D2C sales ledger with planted truth.

Real company ledgers cannot be published, so the benchmark generates 15 months of orders
for a brand selling across regions and channels, then plants what an MIS must find:
one driver per month (a region, channel, category or product that moved against the rest),
unusual days (a bulk order, a site outage), and the junk real exports contain (duplicate
rows, test and cancelled orders, positive returns, other currencies, future dates, blanks).
Truth is kept from the clean ledger *before* the junk is added.

The held-out set changes sizes, weekday pattern and noise, and is exported in a Shopify
layout with Indian formatting (₹, lakh commas, day-first dates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIONS = {"North": 0.32, "West": 0.28, "South": 0.25, "East": 0.15}
CHANNELS = {"Website": 0.45, "Amazon": 0.35, "Retail Stores": 0.20}
CATALOG = {  # category -> [(sku, product, price ₹, cost share)]
    "Skincare": [
        ("SK-01", "Vitamin C Serum", 799, 0.38),
        ("SK-02", "Sunscreen SPF50", 499, 0.42),
        ("SK-03", "Night Cream", 1199, 0.35),
    ],
    "Haircare": [
        ("HC-01", "Onion Hair Oil", 399, 0.45),
        ("HC-02", "Keratin Shampoo", 549, 0.44),
        ("HC-03", "Hair Serum", 699, 0.40),
    ],
    "Wellness": [
        ("WL-01", "Ashwagandha Gummies", 649, 0.50),
        ("WL-02", "Biotin Tablets", 899, 0.48),
        ("WL-03", "Protein Shake", 2199, 0.55),
    ],
    "Bath & Body": [
        ("BB-01", "Body Wash", 349, 0.46),
        ("BB-02", "Body Lotion", 449, 0.43),
        ("BB-03", "Gift Box", 1999, 0.52),
    ],
}
SKUS = [(c, *x) for c, items in CATALOG.items() for x in items]


@dataclass
class Truth:
    monthly: pd.DataFrame  # month -> net_revenue, orders
    drivers: dict[str, tuple[str, str]]  # month -> (dimension, member)
    anomaly_days: list[str]
    junk: dict[str, int] = field(default_factory=dict)


@dataclass
class Profile:
    start: str = "2025-04-01"
    months: int = 15
    orders_per_day: float = 110
    weekday: tuple = (0.9, 0.85, 0.9, 0.95, 1.05, 1.35, 1.3)
    sizes: dict = field(
        default_factory=lambda: {"region": 0.40, "channel": 0.40, "category": 0.45, "product": 0.9}
    )
    n_anomalies: int = 10
    noise: float = 0.08


def _ledger(p: Profile, rng) -> tuple[pd.DataFrame, dict, list]:
    end = pd.Period(p.start, "M") + p.months
    days = pd.date_range(p.start, end.to_timestamp() - pd.Timedelta(days=1), freq="D")
    months = sorted({d.to_period("M") for d in days})
    # one planted driver in every other month of the last 6; the month after it reverses
    events = {}
    dims = ["region", "channel", "category", "product"]
    for i, m in enumerate(months[-6::2]):
        dim = dims[(i + int(rng.integers(0, 4))) % 4]
        member = {
            "region": list(REGIONS),
            "channel": list(CHANNELS),
            "category": list(CATALOG),
            "product": [s[2] for s in SKUS],
        }[dim][int(rng.integers(0, 4 if dim != "channel" else 3))]
        if dim == "product":
            member = SKUS[int(rng.integers(0, len(SKUS)))][2]
        sign = 1 if rng.random() < 0.5 else -1
        events[m] = (dim, member, 1 + sign * p.sizes[dim])
    late = [d for d in days if d >= days[0] + pd.Timedelta(days=70)]
    anomaly_days = sorted(rng.choice(late, p.n_anomalies, replace=False))
    a_mult = {pd.Timestamp(d): (2.6 if rng.random() < 0.5 else 0.3) for d in anomaly_days}

    rows = []
    oid = 100000
    reg, regw = list(REGIONS), np.array(list(REGIONS.values()))
    ch, chw = list(CHANNELS), np.array(list(CHANNELS.values()))
    for d in days:
        m = d.to_period("M")
        doy = d.dayofyear  # festive build-up peaking late October, soft monsoon dip
        season = (
            1
            + 0.30 * np.exp(-(((doy - 300) / 22) ** 2))
            - 0.08 * np.exp(-(((doy - 190) / 25) ** 2))
        )
        trend = 1 + 0.01 * (m - months[0]).n
        lam = p.orders_per_day * p.weekday[d.dayofweek] * season * trend * a_mult.get(d, 1.0)
        lam *= np.exp(rng.normal(0, p.noise))
        ev = events.get(m)
        rw, cw = regw.copy(), chw.copy()
        skw = np.ones(len(SKUS))
        if ev:
            dim, member, mult = ev
            if dim == "region":
                rw[reg.index(member)] *= mult
                lam *= rw.sum() / regw.sum()
            elif dim == "channel":
                cw[ch.index(member)] *= mult
                lam *= cw.sum() / chw.sum()
            elif dim == "category":
                skw[[i for i, s in enumerate(SKUS) if s[0] == member]] *= mult
            else:
                skw[[i for i, s in enumerate(SKUS) if s[2] == member]] *= mult
        n = rng.poisson(lam)
        for _ in range(n):
            oid += 1
            r = reg[rng.choice(4, p=rw / rw.sum())]
            c = ch[rng.choice(3, p=cw / cw.sum())]
            cust = f"cust{int(rng.integers(1, 40000))}@mail.com"
            ts = d + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
            if ev and ev[0] in ("category", "product") and ev[2] > 1:
                lines = 1 + int(rng.random() < 0.35)  # extra demand adds lines, not orders
            else:
                lines = 1 + int(rng.random() < 0.3)
            picks = rng.choice(len(SKUS), size=lines, replace=False, p=skw / skw.sum())
            for k in picks:
                cat, sku, prod, price, cshare = SKUS[k]
                qty = 1 + int(rng.random() < 0.15)
                disc = round(price * qty * float(rng.choice([0, 0, 0.1, 0.15])), 2)
                net = price * qty - disc
                rows.append(
                    [
                        ts,
                        f"#{oid}",
                        cust,
                        sku,
                        prod,
                        cat,
                        r,
                        c,
                        qty,
                        price,
                        disc,
                        net,
                        round(price * qty * cshare, 2),
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
    df = pd.DataFrame(rows, columns=cols)
    # returns: 4% of lines come back 3-15 days later (real, part of the truth)
    ret = df.sample(frac=0.04, random_state=int(rng.integers(1e9))).copy()
    ret["date"] = ret["date"] + pd.to_timedelta(rng.integers(3, 16, len(ret)), unit="D")
    ret = ret[ret["date"] < days[-1]]
    ret["status"] = "return"
    for col in ("qty", "net_amount", "cost"):
        ret[col] = -ret[col]
    df = pd.concat([df, ret]).sort_values("date").reset_index(drop=True)
    return df, events, [str(pd.Timestamp(x).date()) for x in anomaly_days]


def _truth(df: pd.DataFrame, events: dict, anomaly_days: list) -> Truth:
    m = df["date"].dt.to_period("M")
    monthly = pd.DataFrame(
        {
            "net_revenue": df.groupby(m)["net_amount"].sum(),
            "orders": df[df.status == "sale"].groupby(m)["order_id"].nunique(),
        }
    )
    drivers = {}
    for ev_m, (dim, member, _) in events.items():
        drivers[str(ev_m)] = (dim, member)
        drivers[str(ev_m + 1)] = (dim, member)  # the month after: the effect reverses
    return Truth(monthly, drivers, anomaly_days)


def _junk(df: pd.DataFrame, rng, last_day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    junk = {}
    df = df.copy()
    # problems on real rows: returns exported with positive amounts, blank regions
    r = df.index[df.status == "return"]
    flip = rng.choice(r, int(len(r) * 0.6), replace=False)
    for col in ("qty", "net_amount", "cost"):
        df.loc[flip, col] = df.loc[flip, col].abs()
    junk["return_sign"] = len(flip)
    blank = rng.choice(df.index, int(len(df) * 0.01), replace=False)
    df.loc[blank, "region"] = ""
    junk["missing_region"] = len(blank)
    out = [df]
    sale_orders = df.loc[df.status == "sale", "order_id"].unique()
    canc = rng.choice(sale_orders, int(len(sale_orders) * 0.02), replace=False)
    c = df[df.order_id.isin(canc) & (df.status == "sale")].copy()
    c["order_id"] = c["order_id"] + "-C"
    c["status"] = "cancelled"
    out.append(c)
    junk["cancelled"] = len(c)
    t = df[df.status == "sale"].sample(25, random_state=int(rng.integers(1e9))).copy()
    t["customer"] = "test@brand.in"
    t["order_id"] = t["order_id"] + "-T"
    out.append(t)
    junk["test_orders"] = len(t)
    u = df[df.status == "sale"].sample(15, random_state=int(rng.integers(1e9))).copy()
    u["currency"] = "USD"
    u["net_amount"] = (u["net_amount"] / 83).round(2)
    u["order_id"] = u["order_id"] + "-US"
    out.append(u)
    junk["other_currency"] = len(u)
    f = df[df.status == "sale"].sample(6, random_state=int(rng.integers(1e9))).copy()
    f["date"] = last_day + pd.Timedelta(days=200)
    f["order_id"] = f["order_id"] + "-F"
    out.append(f)
    junk["future_date"] = len(f)
    mp = df[df.status == "sale"].sample(12, random_state=int(rng.integers(1e9))).copy()
    mp["net_amount"] = np.nan
    mp["unit_price"] = np.nan
    mp["order_id"] = mp["order_id"] + "-X"
    out.append(mp)
    junk["missing_amount"] = len(mp)
    # the export repeats some real rows exactly
    dup = df.sample(frac=0.015, random_state=int(rng.integers(1e9)))
    out.append(dup)
    junk["duplicate_rows"] = len(dup)
    return pd.concat(out).sort_values("date", kind="stable").reset_index(drop=True), junk


def _indian(x: float) -> str:
    if pd.isna(x):
        return ""
    neg, x = x < 0, abs(x)
    whole, frac = f"{x:.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    s = ",".join(([head] if head else []) + groups + [tail]) if head or groups else tail
    return f"{'-' if neg else ''}₹{s}.{frac}"


def shopify(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows in a Shopify-style layout with Indian formatting."""
    status = df["status"].map({"sale": "paid", "return": "refunded", "cancelled": "voided"})
    out = pd.DataFrame(
        {
            "Name": df["order_id"],
            "Email": df["customer"],
            "Financial Status": status.fillna("paid"),
            "Created at": df["date"].dt.strftime("%d/%m/%Y %H:%M"),
            "Lineitem quantity": df["qty"].abs().astype(int),
            "Lineitem name": df["product"],
            "Lineitem price": df["unit_price"].map(_indian),
            "Lineitem sku": df["sku"],
            "Lineitem discount": df["discount"].map(_indian),
            "Shipping Province": df["region"],
            "Source": df["channel"],
            "Product Type": df["category"],
            "Currency": df["currency"],
            "Total": "",
        }
    )
    # unit price must reproduce the net amount, including USD rows already converted
    usd = df["currency"] == "USD"
    out.loc[usd, "Lineitem price"] = (
        (df.loc[usd, "net_amount"] / df.loc[usd, "qty"]).round(2).astype(str)
    )
    out.loc[usd, "Lineitem discount"] = "0"
    first = ~df["order_id"].duplicated()
    out.loc[first, "Total"] = (
        df.groupby("order_id")["net_amount"].transform("sum")[first].map(_indian)
    )
    # returns in Shopify carry the item price; the sign comes from the status
    return out


def dataset(
    which: str = "dev", seed: int | None = None, months: int | None = None
) -> tuple[pd.DataFrame, Truth, str]:
    """(raw export as it arrives, truth, as_of date). months shortens it for demos and tests."""
    if which == "dev":
        p, s = Profile(), 11 if seed is None else seed
    else:
        p = Profile(
            start="2025-03-01",
            orders_per_day=140,
            weekday=(1.0, 0.95, 0.95, 1.0, 1.1, 1.25, 1.15),
            noise=0.1,
            n_anomalies=8,
            sizes={"region": 0.30, "channel": 0.32, "category": 0.38, "product": 0.75},
        )
        s = 12 if seed is None else seed
    if months:
        p.months = months
        p.n_anomalies = max(2, months)
    rng = np.random.default_rng(s)
    ledger, events, adays = _ledger(p, rng)
    truth = _truth(ledger, events, adays)
    last = ledger["date"].max()
    raw, junk = _junk(ledger, rng, last)
    truth.junk = junk
    if which != "dev":  # Shopify exports every refund line with a positive amount
        truth.junk["return_sign"] = int(((raw.status == "return") & ~raw.duplicated()).sum())
    as_of = str((last + pd.Timedelta(days=1)).date())
    if which != "dev":
        # the Shopify layout has no net amount column: price x qty - discount must rebuild it
        raw = raw.copy()
        ret = raw.status == "return"
        raw.loc[ret, "unit_price"] = (
            raw.loc[ret, "net_amount"].abs() + raw.loc[ret, "discount"]
        ) / raw.loc[ret, "qty"].abs()
        raw = shopify(raw)
    return raw, truth, as_of
