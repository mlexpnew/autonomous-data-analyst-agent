from pathlib import Path

import pytest

from clearmis.synth import dataset


def test_api(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from clearmis.api import app

    raw, _, as_of = dataset("held_out", months=3)
    f = tmp_path / "orders_export.csv"
    raw.to_csv(f, index=False)
    c = TestClient(app)
    r = c.post(f"/v1/reports?as_of={as_of}", files={"file": ("orders_export.csv", f.read_bytes())})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    assert r.json()["summary"].startswith("Net revenue")
    assert c.get(f"/v1/reports/{rid}/xlsx").content[:2] == b"PK"
    a = c.post(f"/v1/reports/{rid}/ask", json={"question": "orders by channel"}).json()
    assert a["ok"]
    assert c.post("/v1/reports", files={"file": ("x.csv", b"foo,bar\n1,2\n")}).status_code == 422


def test_dashboard():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert any(m.label == "Net revenue" for m in at.metric)
