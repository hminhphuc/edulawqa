#!/usr/bin/env python3
"""citation_accuracy.py — Deterministic, judge-free citation metrics for eval200.

For each arm (student and teacher outputs): check the answer's citations against the REAL corpus (Qdrant):
  - cite_exists:  the (normalised so_ky_hieu, dieu) pair exists in the 25,701-chunk corpus
  - cite_support: content-word overlap between the answer and the text of the cited article ≥ 0.4
    (the same threshold as CVRS tier F2 — consistent and pre-existing)
Per-arm metrics: citation precision (exists), support rate, % of answers with ≥1 valid citation, citations/answer.
100% deterministic: no judge, no human. → runs/citation_accuracy/metrics.json

Usage: python3 scripts/citation_accuracy.py
"""
from __future__ import annotations
import argparse
import json
import sys
import re
import unicodedata
from collections import defaultdict
import os
from pathlib import Path

import requests

ROOT = Path(os.environ.get("LEXROUTE_ROOT", Path(__file__).resolve().parent.parent))
# Repo root. Override with the LEXROUTE_ROOT environment variable if you relocate the data.
OUT = ROOT / "runs/citation_accuracy"
QDRANT = "http://localhost:6333"

ARMS = {
    "8B-traj": "runs/student_infer/eval200_8B-traj-f35.jsonl",
    "8B-answer": "runs/student_infer/eval200_8B-answer-f35.jsonl",
    "8B-base": "runs/student_infer/eval200_8B-base.jsonl",
    "4B-traj": "runs/student_infer/eval200_4B-traj-f35.jsonl",
    "4B-answer": "runs/student_infer/eval200_4B-answer-f35.jsonl",
    "4B-base": "runs/student_infer/eval200_4B-base.jsonl",
    "4B-no-route": "runs/student_infer/eval200_4B-no-route-f35.jsonl",
    "4B-unfiltered": "runs/student_infer/eval200_4B-unfiltered-f35.jsonl",
    "teacher-32B": "runs/teacher_anchor/eval200_teacher.jsonl",
}

STOP = set("của và các là được theo trong cho với tại về này đó có không phải người việc quy định điều khoản luật nghị".split())


def norm_doc(s):
    s = unicodedata.normalize("NFC", str(s or "")).upper().strip()
    return re.sub(r"\s+", "", s)


def cwords(s):
    ws = re.findall(r"[\wà-ỹÀ-Ỹ]+", unicodedata.normalize("NFC", str(s or "")).lower())
    return set(w for w in ws if len(w) > 2 and w not in STOP)


def build_corpus_index():
    """(so_ky_hieu_norm, dieu_int) → concatenated text; plus the set of so_ky_hieu present in the corpus."""
    idx = defaultdict(str)
    docs = set()
    off = None
    n = 0
    while True:
        r = requests.post(f"{QDRANT}/collections/legaledu_v3/points/scroll",
                          json={"limit": 1000, "with_payload": ["so_ky_hieu", "dieu", "text"],
                                **({"offset": off} if off else {})}, timeout=60).json()["result"]
        for p in r["points"]:
            pl = p["payload"]
            d = norm_doc(pl.get("so_ky_hieu"))
            docs.add(d)
            dieu = pl.get("dieu")
            m = re.search(r"\d+", str(dieu or ""))
            if m:
                key = (d, int(m.group(0)))
                if len(idx[key]) < 20000:
                    idx[key] += " " + (pl.get("text") or "")
            n += 1
        off = r.get("next_page_offset")
        if not off:
            break
    print(f"corpus index: {n} chunks, {len(idx)} (doc,dieu) keys, {len(docs)} docs")
    return idx, docs


def main():
    ap = argparse.ArgumentParser(
        description="Judge-free citation audit: for each model output, check that every cited "
                    "(document, article) exists in the corpus index and that the answer text "
                    "lexically supports it (support@0.4). Requires the corpus index; the model "
                    "output files evaluated in the paper are not part of this release.")
    ap.add_argument("--arm", action="append", metavar="NAME=PATH",
                    help="model-output JSONL to score, as NAME=PATH (repeatable). "
                         "Default: the paper's eval200 arms under runs/student_infer/, if present.")
    ap.add_argument("--out", default=str(OUT), help="output directory for metrics.json + per-arm files")
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arm) if args.arm else ARMS
    missing = [p for p in arms.values() if not (ROOT / p).exists() and not Path(p).exists()]
    if missing:
        sys.exit("input not found: " + ", ".join(missing[:3]) +
                 "\nPass --arm NAME=PATH pointing at your own model outputs.")
    _run(arms, Path(args.out))


def _run(arms, OUT):
    OUT.mkdir(parents=True, exist_ok=True)
    idx, docs = build_corpus_index()
    art_words = {}  # cache (doc,dieu) -> content words
    metrics = {}
    for arm, path in arms.items():
        rows = [json.loads(l) for l in open((ROOT / path) if (ROOT / path).exists() else path, encoding="utf-8")]
        n_ans = n_cit = n_exist = n_supp = n_ans_valid = 0
        per_q = []
        for r in rows:
            ans = str(r.get("final_answer") or "")
            cits = r.get("citations") or []
            if not ans or not cits:
                per_q.append({"id": r.get("custom_id") or r.get("id"), "n_cit": len(cits),
                              "answered": bool(ans), "any_valid": False})
                continue
            n_ans += 1
            aw = cwords(ans)
            any_valid = False
            for c in cits:
                if not isinstance(c, dict):
                    continue
                n_cit += 1
                key = (norm_doc(c.get("so_ky_hieu")), int(re.search(r"\d+", str(c.get("dieu") or "0")).group(0)) if re.search(r"\d+", str(c.get("dieu") or "")) else -1)
                exists = key in idx
                n_exist += exists
                if exists:
                    any_valid = True
                    if key not in art_words:
                        art_words[key] = cwords(idx[key])
                    ov = len(aw & art_words[key]) / max(len(aw), 1)
                    n_supp += (ov >= 0.4)
            n_ans_valid += any_valid
            per_q.append({"id": r.get("custom_id") or r.get("id"), "n_cit": len(cits),
                          "answered": True, "any_valid": any_valid})
        m = {"n_answers_with_citations": n_ans, "n_citations": n_cit,
             "cite_exists_precision": round(n_exist / max(n_cit, 1), 4),
             "cite_support_rate_0.4": round(n_supp / max(n_cit, 1), 4),
             "answers_with_valid_cite": round(n_ans_valid / max(n_ans, 1), 4),
             "cites_per_answer": round(n_cit / max(n_ans, 1), 2)}
        metrics[arm] = m
        json.dump(per_q, open(OUT / f"perq_{arm}.json", "w"), ensure_ascii=False)
        print(f"[{arm:14s}] answers-w-cite={n_ans:3d} cites={n_cit:4d} "
              f"exists={m['cite_exists_precision']:.1%} support@0.4={m['cite_support_rate_0.4']:.1%} "
              f"≥1valid={m['answers_with_valid_cite']:.1%}")
    json.dump(metrics, open(OUT / "metrics.json", "w"), ensure_ascii=False, indent=1)
    print(f"→ {OUT}/metrics.json")


if __name__ == "__main__":
    main()
