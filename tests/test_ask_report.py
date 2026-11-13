import json

import pandas as pd
import pytest

from clearmis.ask import Plan, ask_question, keyword_plan, llm_plan
from clearmis.report import make, to_excel
from clearmis.synth import dataset


@pytest.fixture(scope="module")
def rep():
    raw, _, as_of = dataset("dev", months=4)
    return make(raw, as_of=as_of)


def test_report_end_to_end(rep, tmp_path):
    assert rep.summary.startswith("Net revenue for")
    json.dumps(rep.as_dict())
    out = to_excel(rep, tmp_path / "mis.xlsx")
    sheets = pd.ExcelFile(out).sheet_names
    assert sheets == [
        "Summary",
        "Monthly KPIs",
        "Drivers",
        "Unusual days",
        "Data fixes",
        "Clean ledger",
    ]


def test_keyword_questions(rep):
    d = rep.clean
    m = str(d["month"].max())
    name = pd.Period(m).strftime("%B %Y")
    r = ask_question(f"Revenue by region for {name}", d)
    assert r["ok"] and set(r["value"]) == set(d.region.unique())
    total = ask_question(f"What was total revenue in {name}?", d)
    assert abs(total["value"] - d[d.month.astype(str) == m].net_amount.sum()) < 0.01
    low = ask_question(f"Which region had the lowest revenue in {name}?", d)
    assert len(low["value"]) == 1
    assert keyword_plan("top 3 products by revenue", d).top == 3


def test_refusals_never_run_code(rep):
    d = rep.clean
    for q in [
        "What is the weather in Delhi?",
        "__import__('os').system('rm -rf /')",
        "revenue for region Atlantis",
    ]:
        assert not ask_question(q, d)["ok"]


def test_llm_plan_is_validated(rep):
    d = rep.clean
    good = llm_plan("x", d, lambda p, t: '{"metric": "orders", "group_by": ["channel"]}')
    assert isinstance(good, Plan) and good.group_by == ["channel"]
    assert llm_plan("x", d, lambda p, t: '{"metric": "orders", "group_by": ["os.system"]}') is None
    assert llm_plan("x", d, lambda p, t: '{"refuse": "not in data"}') is None
    assert llm_plan("x", d, lambda p, t: "import os; os.remove('x')") is None
