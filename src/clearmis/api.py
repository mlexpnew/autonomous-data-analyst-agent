"""HTTP API: upload an export once, then fetch the MIS, the Excel file, or ask questions.

    uvicorn clearmis.api:app
Set CLEARMIS_API_KEYS (comma separated) to require an X-API-Key header.
"""

from __future__ import annotations

import io
import os
import tempfile
import uuid
from collections import OrderedDict

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .ask import ask_question
from .report import Report, make, to_excel

app = FastAPI(
    title="ClearMIS",
    version="1.0.0",
    description="Monthly MIS from raw sales exports, every number verified.",
)
STORE: OrderedDict[str, Report] = OrderedDict()


def auth(x_api_key: str | None = Header(default=None)):
    keys = {k.strip() for k in os.getenv("CLEARMIS_API_KEYS", "").split(",") if k.strip()}
    if keys and x_api_key not in keys:
        raise HTTPException(401, "missing or invalid X-API-Key")


def _get(rid: str) -> Report:
    if rid not in STORE:
        raise HTTPException(404, "unknown report id")
    return STORE[rid]


@app.get("/health")
def health():
    return {"status": "ok", "reports_in_memory": len(STORE)}


@app.post("/v1/reports", dependencies=[Depends(auth)])
async def create(file: UploadFile = File(...), month: str | None = None, as_of: str | None = None):
    data = await file.read()
    name = (file.filename or "upload.csv").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            src = pd.read_excel(io.BytesIO(data))
        else:
            src = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
        rep = make(src, month=month, as_of=as_of)
    except Exception as e:  # bad file, no usable rows
        raise HTTPException(422, f"could not build a report: {e}") from e
    rid = uuid.uuid4().hex[:12]
    STORE[rid] = rep
    while len(STORE) > 50:
        STORE.popitem(last=False)
    return {"id": rid, **rep.as_dict()}


@app.get("/v1/reports/{rid}", dependencies=[Depends(auth)])
def get(rid: str):
    return {"id": rid, **_get(rid).as_dict()}


@app.get("/v1/reports/{rid}/xlsx", dependencies=[Depends(auth)])
def xlsx(rid: str):
    rep = _get(rid)
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as f:
        to_excel(rep, f.name)
        content = open(f.name, "rb").read()
    return Response(
        content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="mis_{rep.month}.xlsx"'},
    )


class Question(BaseModel):
    question: str


@app.post("/v1/reports/{rid}/ask", dependencies=[Depends(auth)])
def ask(rid: str, q: Question):
    return ask_question(q.question, _get(rid).clean)
