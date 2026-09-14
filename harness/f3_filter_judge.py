#!/usr/bin/env python3
"""f3_filter_judge.py — CVRS tier F3: open-weights filter judge scoring trajectories out of 25.

Unlike the evaluation judge (frozen Gemini, which never touches training data), F3 decides
which trajectories ENTER TRAINING, so it must be open-weights. Same /25 rubric with the same
5 criteria (scale kept aligned with the evaluation judge), plus the same abstain-cap ≤18.
Default accept threshold ≥18 (may be relaxed to 16 if the yield is too low; F2 thresholds
are never relaxed).

Usage:
  python3 harness/f3_filter_judge.py --traj PATH/TO/trajectories.jsonl \
      --model Qwen/Qwen3-235B-A22B-Instruct-2507 --tag f3_235b [--limit 30]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import unicodedata
import os
from pathlib import Path

ROOT = Path(os.environ.get("LEXROUTE_ROOT", Path(__file__).resolve().parent.parent))
# Repo root. Override with the LEXROUTE_ROOT environment variable if you relocate the data.
sys.path.insert(0, str(ROOT / "scripts"))
from e04_worker import Worker  # noqa: E402

ABSTAIN_PAT = re.compile(r"không\s+tìm\s+thấy\s+thông\s+tin|không\s+có\s+(đủ\s+)?thông\s+tin|"
                         r"không\s+thể\s+trả\s+lời|chưa\s+đủ\s+(căn\s+cứ|thông\s+tin)")

SYS = """Bạn là giám khảo chấm CHẤT LƯỢNG TRAJECTORY của một agent pháp lý (để lọc data huấn luyện).
Chấm 5 tiêu chí, mỗi tiêu chí 1-5:
1. factual_accuracy — khẳng định khớp với evidence trong steps?
2. citation_quality — citations cụ thể (số ký hiệu + điều), CHỈ từ evidence?
3. reasoning_quality — chuỗi thought → tool_call → answer mạch lạc, tool dùng hợp lý?
4. completeness — answer trả lời đủ câu hỏi?
5. language_quality — tiếng Việt pháp lý chuẩn?
Nguyên tắc: KHÔNG dùng kiến thức luật của riêng bạn để phán evidence sai (chỉ đối chiếu nội bộ
trajectory); answer kiểu "không tìm thấy thông tin" thì tổng hợp lý là 12-18, không cao hơn.
Trả về CHỈ JSON: {"id": "<id đã cho>", "scores": {"factual_accuracy": n, "citation_quality": n,
"reasoning_quality": n, "completeness": n, "language_quality": n}, "total": n, "verdict": "<1 câu>"}"""


def build_prompt(r: dict) -> str:
    steps_txt = []
    for s in r.get("steps", []):
        tc = s.get("tool_call") or {}
        dg = (s.get("tool_result_digest") or {}).get("text", "")
        steps_txt.append(f"[Bước {s['idx']}] thought: {s.get('thought','')[:200]}\n"
                         f"  tool: {tc.get('name')}({json.dumps(tc.get('args', {}), ensure_ascii=False)[:120]})\n"
                         f"  kết quả: {dg[:400]}")
    rd = r.get("route_decision") or {}
    cits = json.dumps(r.get("citations") or [], ensure_ascii=False)[:400]
    return (f"[id: {r['custom_id']}]\n## Câu hỏi: {r['question']}\n"
            f"## Route: {rd.get('label')}/{rd.get('sub')} — {rd.get('rationale','')[:150]}\n"
            f"## Các bước:\n" + "\n".join(steps_txt)[:4000] +
            f"\n## Final answer:\n{(r.get('final_answer') or '(không có)')[:1500]}\n"
            f"## Citations: {cits}\n\nChấm theo rubric, trả JSON.")


def parse(text: str, cid: str) -> dict | None:
    i = 0
    while True:
        s = text.find("{", i)
        if s == -1:
            return None
        depth = 0
        for j in range(s, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(re.sub(r",\s*([}\]])", r"\1", text[s:j + 1]))
                        if "scores" in d and str(d.get("id")) == cid:
                            ssum = sum(float(d["scores"].get(k, 0) or 0) for k in
                                       ["factual_accuracy", "citation_quality", "reasoning_quality",
                                        "completeness", "language_quality"])
                            d["total"] = ssum
                            return d
                    except Exception:
                        pass
                    break
        i = s + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--accept-threshold", type=float, default=18.0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.traj, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    if a.limit:
        rows = rows[:a.limit]
    out_dir = Path(a.traj).parent
    w = Worker(a.model, str(out_dir), max_tokens=800, temperature=0.0,
               extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def build(it):
        return [{"role": "system", "content": SYS}, {"role": "user", "content": build_prompt(it)}]

    def po(text, it):
        d = parse(text, it["custom_id"])
        if d is None:
            return None
        # abstain-cap, aligned with the frozen evaluation judge
        if ABSTAIN_PAT.search(unicodedata.normalize("NFC", (it.get("final_answer") or "")[:150]).casefold()) \
                and d["total"] > 18:
            d["abstain_capped_from"] = d["total"]
            d["total"] = 18.0
        return d

    items = [{**r, "custom_id": r["custom_id"]} for r in rows]
    out_file = str(out_dir / f"{a.tag}.jsonl")
    w.map_items(items, build, po, out_file, max_workers=8)
    recs = [json.loads(l) for l in open(out_file)]
    ok = [r["parsed"] for r in recs if r.get("ok")]
    totals = sorted(r["total"] for r in ok)
    med = totals[len(totals) // 2] if totals else 0
    acc = sum(1 for t in totals if t >= a.accept_threshold)
    print(f"[{a.tag}] n={len(recs)} parse-ok={len(ok)} | median={med}/25 | "
          f"accept(≥{a.accept_threshold})={acc}/{len(ok)} ({100*acc/max(len(ok),1):.0f}%) | cost ${w.cost_usd:.3f}")


if __name__ == "__main__":
    main()
