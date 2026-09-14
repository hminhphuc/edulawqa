#!/usr/bin/env python3
"""kappa_agreement.py — Cohen's κ between human and LLM route labels.

Usage: python3 harness/kappa_agreement.py --human data/annotation/eval200_route_sheet.tsv \
           --gpt data/annotation/gpt_labels.jsonl
Labels: V/G/H (human sheet) vs VECTOR/GRAPH/HYBRID (LLM). kappa computed BEFORE adjudication.
Reports: overall κ, per-class agreement, confusion counts, and the ambiguous rate on each side.
"""
import argparse
import csv
import json
from collections import Counter

MAP = {"V": "VECTOR", "G": "GRAPH", "H": "HYBRID",
       "VECTOR": "VECTOR", "GRAPH": "GRAPH", "HYBRID": "HYBRID"}


def cohens_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--gpt", required=True)
    a = ap.parse_args()
    human = {}
    with open(a.human, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            lab = MAP.get((row.get("ROUTE (V/G/H)") or "").strip().upper())
            if lab:
                human[row["id"]] = {"route": lab,
                                    "sub": (row.get("SUB (T nếu temporal)") or "").strip().upper() == "T",
                                    "amb": (row.get("AMBIGUOUS (x nếu <80% chắc)") or "").strip() != ""}
    gpt = {}
    for l in open(a.gpt, encoding="utf-8"):
        r = json.loads(l)
        if r.get("route"):
            gpt[r["custom_id"]] = {"route": r["route"], "amb": bool(r.get("ambiguous"))}
    common = sorted(set(human) & set(gpt))
    pairs = [(human[i]["route"], gpt[i]["route"]) for i in common]
    k = cohens_kappa(pairs)
    print(f"n chung: {len(common)} (human {len(human)}, gpt {len(gpt)})")
    print(f"Cohen's κ (tổng, TRƯỚC adjudication): {k:.3f}")
    conf = Counter(pairs)
    print("Confusion (human, gpt):", dict(conf))
    for cls in ["VECTOR", "GRAPH", "HYBRID"]:
        sub = [(h, g) for h, g in pairs if h == cls or g == cls]
        agree = sum(1 for h, g in sub if h == g)
        print(f"  {cls}: agree {agree}/{len(sub)}")
    print(f"ambiguous: human {sum(1 for i in common if human[i]['amb'])} | gpt {sum(1 for i in common if gpt[i]['amb'])}")
    print("Pre-registered: kappa mục tiêu ≥0,65; nếu <0,55 → báo thêm gộp nhị phân V vs G∪H")
    if len(pairs) >= 20:
        bin_pairs = [("V" if h == "VECTOR" else "GH", "V" if g == "VECTOR" else "GH") for h, g in pairs]
        print(f"κ nhị phân V vs G∪H: {cohens_kappa(bin_pairs):.3f}")


if __name__ == "__main__":
    main()
