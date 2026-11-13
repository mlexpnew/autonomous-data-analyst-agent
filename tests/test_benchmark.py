from clearmis.evaluate import run


def test_dev_benchmark():
    r = run("dev")
    assert r["kpi"]["clearmis_max_revenue_error_pct"] == 0
    assert r["kpi"]["raw_export_max_revenue_error_pct"] > 5
    assert r["drivers"]["clearmis"] == r["drivers"]["months"]
    assert r["unusual_days"]["clearmis"]["recall"] == 1
    assert r["hallucination_guard"]["unverified"] == 0
    assert all(v["found"] == v["planted"] for v in r["data_quality"].values())
    q = r["questions"]["dev_phrasing"]
    assert q["correct"] == q["questions"]
