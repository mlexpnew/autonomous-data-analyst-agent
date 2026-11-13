"""Command line.

clearmis report sales.csv --out mis.xlsx         # monthly MIS for the latest complete month
clearmis report sales.csv --month 2026-05
clearmis ask sales.csv "revenue by region last month"
clearmis demo --out sample_sales.csv            # write a sample Shopify-style export
"""

from __future__ import annotations

import argparse
import os

from .ask import ask_question
from .report import make, to_excel


def _rewrite():
    provider = os.getenv("CLEARMIS_LLM")
    if not provider:
        return None, None
    from .llm import chat, rewriter

    ask = chat(provider)
    return rewriter(ask), ask


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clearmis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("file")
    r.add_argument("--month")
    r.add_argument("--out", default="mis_report.xlsx")
    a = sub.add_parser("ask")
    a.add_argument("file")
    a.add_argument("question")
    dm = sub.add_parser("demo")
    dm.add_argument("--out", default="sample_sales.csv")
    dm.add_argument("--months", type=int, default=6)
    args = ap.parse_args(argv)
    rewrite, ask = _rewrite()

    if args.cmd == "demo":
        from .synth import dataset

        raw, _, _ = dataset("held_out", months=args.months)
        raw.to_csv(args.out, index=False)
        print(f"wrote {len(raw):,} rows to {args.out}")
        return 0
    rep = make(args.file, month=getattr(args, "month", None), rewrite=rewrite)
    if args.cmd == "report":
        to_excel(rep, args.out)
        print(rep.summary)
        for line in rep.dropped_sentences:
            print("  (dropped, unverified):", line)
        print(f"\n-> {args.out}")
        return 0
    res = ask_question(args.question, rep.clean, ask)
    print(res["answer"])
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
