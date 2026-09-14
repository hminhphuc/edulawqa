#!/usr/bin/env python3
"""e37_nli_citation.py — NLI cross-check for citation support (addresses the "extractiveness" concern).

Motivation: "support@0.4 is lexical overlap, so it may reward verbatim copying and measure
extractiveness rather than grounding — ALCE uses NLI for exactly this reason." Cross-check:
re-score EVERY corpus-existing citation of the 4 F3.5 arms (eval200) with GPT-5-mini entailment
(does the article support the answer content?), then compare the ORDERING with lexical
support@0.4 (scripts/citation_accuracy.py). If the ordering holds (traj >> answer under both
constructs) the gap is real grounding, not an artifact of the construct. Fabrication
(cite_exists) is reported separately — it cannot be gamed by copying.

Batch-safety protocol: a unique custom_id per request (arm|qid|cit_idx); the LLM echoes the id
in its JSON; results are re-aligned by id; the id-match rate is logged; broken records go to
bad_records.jsonl (no silent skipping).

Usage: python3 scripts/e37_nli_citation.py [--dry-run]   (needs OPENAI_API_KEY in the environment)
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as rq

ROOT = Path(os.environ.get("LEXROUTE_ROOT", Path(__file__).resolve().parent.parent))
# Repo root. Override with the LEXROUTE_ROOT environment variable if you relocate the data.
sys.path.insert(0, str(ROOT / "scripts"))
from citation_accuracy import build_corpus_index, norm_doc  # noqa: E402

OUT = ROOT / "runs/E37_nli-citation"
ARMS = {
    "8B-traj": "runs/student_infer/eval200_8B-traj-f35.jsonl",
    "8B-answer": "runs/student_infer/eval200_8B-answer-f35.jsonl",
    "4B-traj": "runs/student_infer/eval200_4B-traj-f35.jsonl",
    "4B-answer": "runs/student_infer/eval200_4B-answer-f35.jsonl",
}
MODEL = "gpt-5-mini"

PROMPT = """Bạn là chuyên gia thẩm định trích dẫn pháp lý. Cho MỘT điều luật và MỘT câu trả lời của hệ thống QA, hãy đánh giá: nội dung pháp lý chính của CÂU TRẢ LỜI có được ĐIỀU LUẬT này chống đỡ (entail/support) không?

- SUPPORTED: các khẳng định pháp lý chính của câu trả lời được điều luật chống đỡ trực tiếp (kể cả khi diễn đạt khác — paraphrase vẫn tính).
- PARTIAL: điều luật chống đỡ một phần nội dung, phần còn lại không có trong điều luật này.
- NOT_SUPPORTED: điều luật không liên quan hoặc mâu thuẫn với nội dung trả lời.

ĐIỀU LUẬT ({doc} — Điều {dieu}):
{article}

CÂU HỎI: {question}

CÂU TRẢ LỜI CẦN THẨM ĐỊNH:
{answer}

Trả về JSON duy nhất, không thêm text:
{{"custom_id": "{cid}", "verdict": "SUPPORTED|PARTIAL|NOT_SUPPORTED"}}"""


def build_requests(idx):
    reqs = []
    for arm, path in ARMS.items():
        for r in (ROOT / path).open(encoding="utf-8"):
            r = json.loads(r)
            ans = str(r.get("final_answer") or "")
            cits = r.get("citations") or []
            if not ans or not cits:
                continue
            qid = r.get("custom_id") or r.get("id")
            for ci, c in enumerate(cits):
                if not isinstance(c, dict):
                    continue
                m = re.search(r"\d+", str(c.get("dieu") or ""))
                key = (norm_doc(c.get("so_ky_hieu")), int(m.group(0)) if m else -1)
                if key not in idx:
                    continue  # fabricated citations are measured separately by cite_exists (citation_accuracy.py)
                cid = f"{arm}|{qid}|{ci}"
                reqs.append({"custom_id": cid, "arm": arm, "qid": qid,
                             "prompt": PROMPT.format(doc=key[0], dieu=key[1],
                                                     article=idx[key][:3500],
                                                     question=str(r.get("question") or "")[:500],
                                                     answer=ans[:2000], cid=cid)})
    return reqs


def call_one(req, api_key):
    for att in range(3):
        try:
            resp = rq.post("https://api.openai.com/v1/chat/completions",
                           headers={"Authorization": f"Bearer {api_key}"},
                           json={"model": MODEL,
                                 "messages": [{"role": "user", "content": req["prompt"]}],
                                 "max_completion_tokens": 2000},
                           timeout=120)
            resp.raise_for_status()
            j = resp.json()
            text = j["choices"][0]["message"]["content"] or ""
            usage = j.get("usage", {})
            m = re.search(r"\{.*\}", text, re.S)
            d = json.loads(m.group(0)) if m else {}
            return {"custom_id_sent": req["custom_id"], "custom_id_echo": d.get("custom_id"),
                    "verdict": d.get("verdict"), "raw": text[:200],
                    "tok_in": usage.get("prompt_tokens", 0), "tok_out": usage.get("completion_tokens", 0)}
        except Exception as e:
            if att == 2:
                return {"custom_id_sent": req["custom_id"], "custom_id_echo": None,
                        "verdict": None, "raw": f"ERROR: {e}", "tok_in": 0, "tok_out": 0}
            time.sleep(3 * (att + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx, _ = build_corpus_index()
    reqs = build_requests(idx)
    est_in = sum(len(r["prompt"]) for r in reqs) // 4
    est_cost = est_in / 1e6 * 0.25 + len(reqs) * 60 / 1e6 * 2.0
    print(f"requests={len(reqs)} | est_in_tokens≈{est_in:,} | est_cost≈${est_cost:.2f} ({MODEL})")
    if a.dry_run:
        for arm in ARMS:
            print(f"  {arm}: {sum(1 for r in reqs if r['arm'] == arm)}")
        return

    api_key = os.environ["OPENAI_API_KEY"]
    raw_f = (OUT / "raw_responses.jsonl").open("w", encoding="utf-8")
    bad_f = (OUT / "bad_records.jsonl").open("w", encoding="utf-8")
    results = {}
    n_bad = 0
    tok_in = tok_out = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(call_one, r, api_key): r for r in reqs}
        for i, fut in enumerate(as_completed(futs), 1):
            out = fut.result()
            raw_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            tok_in += out["tok_in"]; tok_out += out["tok_out"]
            # RE-ALIGN BY ECHOED ID — never by arrival order
            if out["custom_id_echo"] and out["verdict"] in ("SUPPORTED", "PARTIAL", "NOT_SUPPORTED"):
                results[out["custom_id_echo"]] = out["verdict"]
            else:
                n_bad += 1
                bad_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(reqs)} ({time.time()-t0:.0f}s)", flush=True)
    raw_f.close(); bad_f.close()

    sent_ids = {r["custom_id"] for r in reqs}
    matched = len(sent_ids & set(results))
    match_rate = matched / len(sent_ids)
    print(f"\nsent={len(sent_ids)} | id-matched={matched} | match_rate={match_rate:.3f} | bad={n_bad}")
    if match_rate < 0.90:
        print("⚠️ match_rate < 90% — kiểm tra bad_records trước khi dùng!")

    # per-arm metrics + comparison with the lexical metric
    lex = json.load(open(ROOT / "runs/E25_citation-accuracy/metrics.json")) if (ROOT / "runs/E25_citation-accuracy/metrics.json").exists() else {}
    metrics = {}
    print(f"\n{'arm':10}{'n_cit':>6}{'NLI-full':>10}{'NLI-full+part':>14}{'lexical@0.4':>12}")
    for arm in ARMS:
        vs = [v for k, v in results.items() if k.startswith(arm + "|")]
        n = len(vs)
        full = sum(v == "SUPPORTED" for v in vs) / max(n, 1)
        fp = sum(v in ("SUPPORTED", "PARTIAL") for v in vs) / max(n, 1)
        lx = (lex.get(arm) or {}).get("cite_support_rate_0.4")
        metrics[arm] = {"n_scored": n, "nli_supported": round(full, 3),
                        "nli_supported_or_partial": round(fp, 3), "lexical_support_0.4": lx}
        print(f"{arm:10}{n:>6}{full:>10.3f}{fp:>14.3f}{str(lx):>12}")
    cost = tok_in / 1e6 * 0.25 + tok_out / 1e6 * 2.0
    meta = {"model": MODEL, "n_requests": len(reqs), "id_match_rate": round(match_rate, 4),
            "n_bad": n_bad, "tok_in": tok_in, "tok_out": tok_out,
            "cost_actual_usd": round(cost, 3), "cost_est_usd": round(est_cost, 3)}
    json.dump({"meta": meta, "metrics": metrics}, open(OUT / "nli_metrics.json", "w"),
              ensure_ascii=False, indent=1)
    (OUT / "config.yaml").write_text(
        f"run_id: E37_nli-citation\nmodel: {MODEL}\nn_requests: {len(reqs)}\n"
        f"id_match_rate: {match_rate:.4f}\ncost_est_usd: {est_cost:.3f}\ncost_actual_usd: {cost:.3f}\n"
        f"purpose: NLI cross-check citation-support (extractiveness rebuttal, C9)\n")
    print(f"\ncost actual=${cost:.3f} (est ${est_cost:.2f}) → {OUT}/nli_metrics.json")


if __name__ == "__main__":
    main()
