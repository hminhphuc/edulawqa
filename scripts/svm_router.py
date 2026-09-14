#!/usr/bin/env python3
"""svm_router.py — TF-IDF + LinearSVC route classifier (must-beat baseline; cf. the 93.2% SVM baseline in RAGRouter-Bench).

Trained on the full silver pool.
3 route classes {VECTOR, GRAPH, HYBRID}; REFUSE is excluded (an answer-layer behaviour, not a route).
Evaluate on EduLawQA-Route reference labels; also reports 5-fold CV on the silver pool and saves the model.
No leakage: pool ∩ eval200 = ∅ (two-stage dedup). The disjointness evidence is printed by --train.

Usage: python3 scripts/svm_router.py --train        (train + 5-fold CV on the silver pool)
       python3 scripts/svm_router.py --eval <gold_routes.tsv>   (score on the eval200 gold labels)
"""
from __future__ import annotations
import argparse
import json
import pickle
import os
from pathlib import Path

ROOT = Path(os.environ.get("LEXROUTE_ROOT", Path(__file__).resolve().parent.parent))
# Repo root. Override with the LEXROUTE_ROOT environment variable if you relocate the data.
POOL = ROOT / "data/train_pool/pool_v1_silver.jsonl"
MODEL_P = ROOT / "runs/svm_router/model.pkl"
ROUTES = ["VECTOR", "GRAPH", "HYBRID"]


def load_pool():
    X, y = [], []
    for l in POOL.open(encoding="utf-8"):
        r = json.loads(l)
        lab = (r.get("silver_route") or {}).get("label")
        if lab in ROUTES:
            X.append(r["question"])
            y.append(lab)
    return X, y


def build_pipe():
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000)),
        ])),
        ("svm", LinearSVC(C=1.0, class_weight="balanced")),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval")
    a = ap.parse_args()
    MODEL_P.parent.mkdir(parents=True, exist_ok=True)

    if a.train:
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import classification_report, f1_score
        X, y = load_pool()
        print(f"train: {len(X)} câu silver | dist {({r: y.count(r) for r in ROUTES})}")
        pipe = build_pipe()
        pred = cross_val_predict(pipe, X, y, cv=5)
        print("5-fold CV trên silver (sanity — KHÔNG phải số paper):")
        print(classification_report(y, pred, digits=3))
        print(f"macro-F1 CV: {f1_score(y, pred, average='macro'):.3f}")
        pipe.fit(X, y)
        with MODEL_P.open("wb") as f:
            pickle.dump(pipe, f)
        # disjointness evidence vs eval200
        import unicodedata, re
        def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).casefold()).strip()
        eval_qs = {norm(json.loads(l)["question"]) for l in (ROOT / "eval/data/questions_200.jsonl").open() if l.strip()}
        overlap = sum(1 for q in X if norm(q) in eval_qs)
        print(f"disjoint check: pool ∩ eval200 = {overlap} (phải 0) → {'✓' if overlap==0 else '🔴 LEAK'}")
        print(f"model → {MODEL_P}")

    if a.eval:
        import csv
        from sklearn.metrics import classification_report, f1_score, accuracy_score
        with MODEL_P.open("rb") as f:
            pipe = pickle.load(f)
        gold, qs = {}, {}
        for l in (ROOT / "eval/data/questions_200.jsonl").open():
            if l.strip():
                r = json.loads(l)
                qs[r["id"]] = r["question"]
        with open(a.eval, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                lab = {"V": "VECTOR", "G": "GRAPH", "H": "HYBRID"}.get((row.get("ROUTE (V/G/H)") or "").strip().upper())
                amb = (row.get("AMBIGUOUS (x nếu <80% chắc)") or "").strip()
                if lab and not amb:  # exclude ambiguous items from routing accuracy
                    gold[row["id"]] = lab
        ids = [i for i in gold if i in qs]
        pred = pipe.predict([qs[i] for i in ids])
        yt = [gold[i] for i in ids]
        print(f"SVM-full trên eval200 gold (loại ambiguous): n={len(ids)}")
        print(f"  accuracy {accuracy_score(yt, pred):.3f} | macro-F1 {f1_score(yt, pred, average='macro'):.3f}")
        print(classification_report(yt, pred, digits=3))
        print("  (must-beat: learned-router phải ≥ số này — C2/P3)")


if __name__ == "__main__":
    main()
