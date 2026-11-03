import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
CUT = pd.Timestamp("2026-09-22")                 # the TermTrove copy and every archive date stop here

PAPER_NAME = {
    "Core DeWitt Clause": "Benchmarking and Results Restrictions",
    "DeWitt Embrace Clause": "Conditional Benchmarking Permissions",
    "Competitive Ban": "Competitive Analysis Restrictions",
    "Publicity Gag Clause": "Service-Related Publicity Restrictions",
    "Confidential Information Restriction": "Evaluation-Related Confidentiality Restrictions",
}
LABELS = list(PAPER_NAME)
DB_NAME = {v: k for k, v in PAPER_NAME.items()}


def _read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def documents(runs=None):
    """Documents with one bool column per label (clause in the latest version). runs: keep only clauses from these runs."""
    rows = _read("documents.jsonl")
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "clauses"} for r in rows])
    for l in LABELS:
        d[l] = [any(DB_NAME[c["label"]] == l and (runs is None or c["run"] in runs) for c in r["clauses"]) for r in rows]
    return d


def history(sources=("wayback", "termtrove")):
    """Archived observations (dated <= CUT) with one bool column per label."""
    rows = [r for r in _read("history.jsonl") if r["source"] in sources]
    h = pd.DataFrame(rows)
    for l in LABELS:
        h[l] = [PAPER_NAME[l] in r["labels"] for r in rows]
    h["date"] = pd.to_datetime(h.date)
    return h.drop(columns="labels")


def wayback_years(docs):
    """Document x year: does any Wayback snapshot that year carry each clause?"""
    h = history(("wayback",))
    h = h[h.document_id.isin(set(docs))]
    h["year"] = h.date.dt.year
    return h


def capiq():
    p = os.path.join(DATA, "capiq", "capiq_company_info.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Capital IQ file not found: {p}\nThe industry and revenue outputs need it; it cannot be redistributed.")
    ci = pd.read_csv(p, encoding="utf-8-sig", dtype={"spciq_id": str}, usecols=["spciq_id", "iq_industry", "iq_total_rev"],
                     keep_default_na=False, na_values=[""])                # "NA" etc. stay text
    return ci.rename(columns={"iq_total_rev": "revenue"})               # total revenue, thousands of US dollars


def with_industry(d):
    """Add sector / industry / matched from Capital IQ (via the document's spciq_id)."""
    ci = capiq()[["spciq_id", "iq_sector", "iq_industry"]].rename(columns={"iq_sector": "sector", "iq_industry": "industry"})
    d = d.merge(ci, on="spciq_id", how="left")
    d["matched"] = d.spciq_id.notna()
    return d


def gics_label(name):
    """Capital IQ spells GICS names with "and"; the official GICS names use "&" (and are shorter)."""
    name = {"Electronic Equipment, Instruments and Components": "Electronic Equipment and Instruments"}.get(name, name)
    return name.replace(" and ", r" \& ")


def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def save(fig, path, **kw):
    fig.savefig(path, **kw)
    return path
