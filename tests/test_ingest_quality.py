import numpy as np
import pandas as pd

from clearmis.ingest import load, map_columns, normalise_status, parse_dates, parse_number
from clearmis.quality import clean


def test_column_mapping_presets():
    shopify = [
        "Name",
        "Email",
        "Financial Status",
        "Created at",
        "Lineitem quantity",
        "Lineitem name",
        "Lineitem price",
        "Lineitem sku",
        "Shipping Province",
        "Total",
    ]
    m = map_columns(shopify)
    assert m["Name"] == "order_id" and m["Lineitem price"] == "unit_price"
    assert "Total" not in m  # order-level total must not be read as a line amount
    tally = [
        "Date",
        "Voucher No.",
        "Particulars",
        "Stock Item",
        "Quantity",
        "Rate",
        "Value",
        "Voucher Type",
    ]
    t = map_columns(tally)
    assert (
        t["Value"] == "net_amount"
        and t["Particulars"] == "customer"
        and t["Voucher Type"] == "status"
    )


def test_indian_numbers_and_dates():
    assert parse_number("₹1,23,456.50") == 123456.5
    assert parse_number("Rs. 2,500") == 2500
    assert parse_number("(1,200.00)") == -1200
    assert np.isnan(parse_number("-"))
    d = parse_dates(pd.Series(["05/03/2026", "2026-03-05", "05-03-2026 14:00"]))
    assert all(x.month == 3 and x.day == 5 for x in d)
    assert [normalise_status(s) for s in ["paid", "Refunded", "voided", "Credit Note", ""]] == [
        "sale",
        "return",
        "cancelled",
        "return",
        "sale",
    ]


def _raw():
    rows = [
        [
            "2026-05-01",
            "#1",
            "a@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "North",
            "Web",
            1,
            500,
            0,
            500,
            200,
            "paid",
            "INR",
        ],
        [
            "2026-05-01",
            "#1",
            "a@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "North",
            "Web",
            1,
            500,
            0,
            500,
            200,
            "paid",
            "INR",
        ],
        [
            "2026-05-02",
            "#2",
            "test@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "North",
            "Web",
            1,
            500,
            0,
            500,
            200,
            "paid",
            "INR",
        ],
        [
            "2026-05-03",
            "#3",
            "b@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "",
            "Web",
            2,
            500,
            0,
            1000,
            400,
            "paid",
            "INR",
        ],
        [
            "2026-05-04",
            "#4",
            "c@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "South",
            "Web",
            1,
            500,
            0,
            500,
            200,
            "voided",
            "INR",
        ],
        [
            "2026-05-06",
            "#3",
            "b@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "South",
            "Web",
            1,
            500,
            0,
            500,
            200,
            "refunded",
            "INR",
        ],
        [
            "2026-05-07",
            "#5",
            "d@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "South",
            "Web",
            1,
            6,
            0,
            6,
            2,
            "paid",
            "USD",
        ],
        [
            "2027-01-01",
            "#6",
            "e@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "South",
            "Web",
            1,
            500,
            0,
            500,
            200,
            "paid",
            "INR",
        ],
        [
            "2026-05-08",
            "#7",
            "f@x.in",
            "SK-1",
            "Serum",
            "Skin",
            "South",
            "Web",
            1,
            "",
            0,
            "",
            "",
            "paid",
            "INR",
        ],
    ]
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
    return pd.DataFrame(rows, columns=cols).astype(str)


def test_cleaning_counts_every_fix():
    raw, _ = load(_raw())
    d, issues = clean(raw, as_of=pd.Timestamp("2026-06-01"))
    got = {i.code: i.rows for i in issues}
    assert got == {
        "duplicate_rows": 1,
        "missing_amount": 1,
        "test_orders": 1,
        "cancelled": 1,
        "future_date": 1,
        "other_currency": 1,
        "return_sign": 1,
        "missing_region": 1,
    }
    assert d["net_amount"].sum() == 500 + 1000 - 500
    assert (d.region == "Unknown").sum() == 1
    fx, _ = clean(raw, as_of=pd.Timestamp("2026-06-01"), fx={"USD": 83})
    assert fx["net_amount"].sum() == 1000 + 6 * 83
