#!/usr/bin/env python3
"""silver_labels.py — Silver route labels for the training pool.

Two sources: (1) generation intent (the route the question was GENERATED for — a strong prior);
             (2) an INDEPENDENT Qwen3-32B classification following the codebook's two-gate procedure.
Agree → silver-high. Disagree → take the Qwen label if its rationale names a relation/content
(a pre-registered machine rule — the ~390 disagreements are NOT all sent to humans), tagged
silver-low; silver-low items are EXCLUDED from F4 route enforcement (the teacher may override
freely). Humans review only a 50-item sample of disagreements to measure the rule's error rate
(reported in the paper).

Usage: python3 scripts/silver_labels.py --pool data/train_pool/pool_v1.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
import os
from pathlib import Path

ROOT = Path(os.environ.get("LEXROUTE_ROOT", Path(__file__).resolve().parent.parent))
# Repo root. Override with the LEXROUTE_ROOT environment variable if you relocate the data.
sys.path.insert(0, str(ROOT / "scripts"))
from e04_worker import Worker  # noqa: E402

SYS = """Bạn gán nhãn ROUTE cho câu hỏi pháp luật giáo dục theo quy trình 2 CỔNG (dừng ở nhãn đầu):
G1: Trả lời ĐÚNG-ĐỦ có BẮT BUỘC đi qua ≥1 quan hệ liên-văn-bản tường minh (sửa đổi/bổ sung,
    thay thế/bãi bỏ, hiệu lực, hướng dẫn/căn cứ, viện dẫn chéo) giữa ≥2 văn bản không?
    KHÔNG → VECTOR. CÓ → G2.
G2: Ngoài quan hệ đó có cần trích NỘI DUNG nguyên văn điều khoản không? KHÔNG → GRAPH. CÓ → HYBRID.
Sub TEMPORAL: quan hệ thuộc {hiệu lực, sửa đổi, thay thế, bãi bỏ}.
Câu kiểu phải TỪ CHỐI (ngoài miền/văn bản bịa) → nhãn REFUSE.
Trả CHỈ JSON: {"id": "<id>", "route": "VECTOR|GRAPH|HYBRID|REFUSE", "sub": "TEMPORAL"|null,
"rationale": "<≤25 từ nêu G1/G2>"}"""

REL_WORDS = ("quan hệ", "sửa đổi", "thay thế", "bãi bỏ", "hiệu lực", "hướng dẫn", "căn cứ",
             "dẫn chiếu", "liên văn bản", "hai văn bản", "g1", "g2", "nội dung", "một văn bản",
             "ngoài miền", "không tồn tại", "bịa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/train_pool/pool_v1.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.pool, encoding="utf-8")]
    w = Worker("Qwen/Qwen3-32B", str(ROOT / "runs/QGEN/silver"), max_tokens=300, temperature=0.0,
               cost_cap_usd=8.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def build(it):
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": f"[id: {it['custom_id']}]\nCâu hỏi: {it['question']}"}]

    def po(text, it):
        i = 0
        while True:
            s = text.find("{", i)
            if s == -1:
                return None
            d0 = 0
            for j in range(s, len(text)):
                if text[j] == "{":
                    d0 += 1
                elif text[j] == "}":
                    d0 -= 1
                    if d0 == 0:
                        try:
                            d = json.loads(text[s:j + 1])
                            if str(d.get("id")) == it["custom_id"] and \
                                    str(d.get("route", "")).upper() in ("VECTOR", "GRAPH", "HYBRID", "REFUSE"):
                                return {"route": d["route"].upper(),
                                        "sub": d.get("sub") if d.get("sub") not in ("null", "") else None,
                                        "rationale": str(d.get("rationale", ""))[:200]}
                        except Exception:
                            pass
                        break
            i = s + 1

    out_file = str(ROOT / "runs/QGEN/silver/classify.jsonl")
    w.map_items(rows, build, po, out_file, max_workers=10)

    cls = {json.loads(l)["custom_id"]: (json.loads(l).get("parsed") or {})
           for l in open(out_file, encoding="utf-8")}
    # merge by the machine rule
    INTENT_MAP = {"VECTOR": "VECTOR", "GRAPH_NONT": "GRAPH", "GRAPH_T": "GRAPH",
                  "HYBRID": "HYBRID", "REFUSE": "REFUSE"}
    stats = Counter()
    out = []
    for r in rows:
        intent = INTENT_MAP.get(r.get("silver_route_intent", ""), "?")
        q = cls.get(r["custom_id"]) or {}
        qlab = q.get("route")
        sub_intent = "TEMPORAL" if r.get("silver_route_intent") == "GRAPH_T" else None
        if intent == "REFUSE" or r.get("origin") == "template":
            # constructed ground truth: refuse questions are BUILT from fabricated/out-of-domain
            # document numbers (the classifier has no corpus access, so it cannot tell); template
            # items carry a machine-generated answer_gt → intent wins
            lab, conf, sub = intent, "silver-high-constructed", sub_intent
        elif not qlab:
            lab, conf, sub = intent, "silver-mid-noclassify", sub_intent
        elif qlab == intent:
            lab, conf = intent, "silver-high"
            sub = q.get("sub") or sub_intent
        else:
            rat = (q.get("rationale") or "").casefold()
            rationale_valid = any(k in rat for k in REL_WORDS)
            if rationale_valid:
                lab, conf, sub = qlab, "silver-low", q.get("sub")
            else:
                lab, conf, sub = intent, "silver-low", sub_intent
        stats[conf] += 1
        out.append({**r, "silver_route": {"label": lab, "sub": sub, "confidence": conf,
                                          "intent": intent, "classify": qlab,
                                          "classify_rationale": q.get("rationale")}})
    outp = ROOT / "data/train_pool/pool_v1_silver.jsonl"
    with outp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = Counter(r["silver_route"]["label"] for r in out)
    print(f"silver: {dict(stats)} | nhãn cuối: {dict(dist)} | cost ${w.cost_usd:.2f}")
    # 50-item disagreement sample for humans to measure the rule's error rate
    dis = [r for r in out if r["silver_route"]["confidence"] == "silver-low"][:50]
    dp = ROOT / "data/annotation/disagree_sample_50.jsonl"
    with dp.open("w", encoding="utf-8") as f:
        for r in dis:
            f.write(json.dumps({"custom_id": r["custom_id"], "question": r["question"],
                                "intent": r["silver_route"]["intent"],
                                "classify": r["silver_route"]["classify"],
                                "final": r["silver_route"]["label"]}, ensure_ascii=False) + "\n")
    print(f"→ {outp} | disagree sample: {dp} ({len(dis)} câu cho người duyệt đo error-rate)")


if __name__ == "__main__":
    main()
