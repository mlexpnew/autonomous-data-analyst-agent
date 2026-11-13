"""ClearMIS dashboard.  streamlit run app.py"""

from __future__ import annotations

import io
import tempfile

import pandas as pd
import streamlit as st

from clearmis.ask import ask_question
from clearmis.narrative import inr
from clearmis.report import make, to_excel
from clearmis.synth import dataset

st.set_page_config(page_title="ClearMIS", page_icon="📊", layout="wide")
st.title("ClearMIS")
st.caption(
    "Your monthly MIS from a raw sales export: cleaned, explained, and every number checked."
)

up = st.sidebar.file_uploader("Sales export (CSV or Excel)", type=["csv", "xlsx", "xls"])
st.sidebar.caption(
    "Shopify, Tally sales register, marketplace reports or any sheet with dates and amounts."
)


@st.cache_data(show_spinner="Building the sample export...")
def sample() -> tuple[pd.DataFrame, str]:
    raw, _, as_of = dataset("held_out", months=6)
    return raw, as_of


if up is None:
    st.info("No file uploaded: showing a 6-month sample export from a D2C brand (Shopify layout).")
    src, as_of = sample()
else:
    data = up.getvalue()
    src = (
        pd.read_excel(io.BytesIO(data))
        if up.name.lower().endswith(("xlsx", "xls"))
        else pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    )
    as_of = None

rep = make(src, as_of=as_of)
months = [str(m) for m in rep.kpis.index]
pick = st.sidebar.selectbox("Month", months[1:][::-1] or months)
if pick != str(rep.month):
    rep = make(src, month=pick, as_of=as_of)

k = rep.kpis.loc[rep.month]
c = st.columns(5)
c[0].metric(
    "Net revenue", inr(k.net_revenue), f"{k.mom_pct:+.1f}%" if pd.notna(k.mom_pct) else None
)
c[1].metric("Orders", f"{int(k.orders):,}")
c[2].metric("Average order", inr(k.aov))
c[3].metric("Return rate", f"{k.return_rate_pct:.1f}%")
if "gross_margin_pct" in k and pd.notna(k.gross_margin_pct):
    c[4].metric("Gross margin", f"{k.gross_margin_pct:.1f}%")

st.subheader(f"Summary for {rep.month.strftime('%B %Y')}")
st.write(rep.summary)
st.caption("Every number above is checked against the computed results before it is shown.")

left, right = st.columns(2)
with left:
    st.subheader("Net revenue by month")
    chart = rep.kpis["net_revenue"].copy()
    chart.index = chart.index.astype(str)
    st.bar_chart(chart)
with right:
    st.subheader("What drove the change")
    st.dataframe(
        pd.DataFrame([d.as_dict() for d in rep.drivers]).round(0),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("Unusual days")
an = pd.DataFrame([a.as_dict() for a in rep.anomalies])
st.dataframe(an.round(1) if len(an) else an, hide_index=True, use_container_width=True)

st.subheader("Data fixed before reporting")
st.dataframe(
    pd.DataFrame([i.as_dict() for i in rep.issues]), hide_index=True, use_container_width=True
)

st.subheader("Ask a question")
q = st.text_input("For example: revenue by region last month, top 3 products in May 2026")
if q:
    res = ask_question(q, rep.clean)
    (st.success if res["ok"] else st.warning)(res["answer"])

with tempfile.NamedTemporaryFile(suffix=".xlsx") as f:
    to_excel(rep, f.name)
    st.download_button(
        "Download the MIS as Excel", open(f.name, "rb").read(), file_name=f"mis_{rep.month}.xlsx"
    )
