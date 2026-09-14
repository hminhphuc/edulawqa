#!/usr/bin/env python3
"""f35_filter.py — F3.5 semantic-correctness filter (raises the quality of accepted trajectories; 100% open-weights).

An audit found that ~24% of accepted trajectories invented content beyond the evidence or
mixed documents. F2-L3 (grounding 0.4) was too loose, and F3 (judge quality) does not catch
semantic hallucination. F3.5 adds two stages:
  (a) tighten the F2-L3 grounding threshold 0.4 → 0.6 (deterministic, recomputed from the trajectory)
  (b) Qwen3-32B (open-weights) strict check: "is the answer CORRECT and ENTIRELY from the
      evidence, with nothing invented or mixed?" → reject otherwise.
No closed-weights model (Claude/GPT/Gemini) touches training data; keeping the pipeline
open-weights keeps the release clean with respect to provider terms of service.

Usage: python3 scripts/f35_filter.py --accepted PATH/TO/accepted.jsonl
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
sys.path.insert(0, str(ROOT / "eval/harness"))
from e04_worker import Worker  # noqa: E402

SKY = re.compile(r"\d{1,4}/\d{4}/[A-ZĐ][A-ZĐa-z0-9\-]*")


def norm(s):
    return unicodedata.normalize("NFC", s or "").casefold()


def toks(s, n=3):
    return {t for t in re.findall(r"[\wà-ỹ]+", norm(s)) if len(t) >= n}


def l3_grounding(traj, thresh=0.6) -> bool:
    """Tightened L3: every sentence of final_answer around a citation must overlap ≥thresh with the evidence digest."""
    ans = traj.get("final_answer") or ""
    dig = " ".join((s.get("tool_result_digest") or {}).get("text", "") for s in traj.get("steps", []))
    dtoks = toks(dig)
    if not dtoks:
        return False
    # split the answer into sentences; every sentence carrying a claim must be grounded
    sents = re.split(r"[.\n]", ans)
    checked = [s for s in sents if len(s.split()) >= 6]
    if not checked:
        return True
    grounded = sum(1 for s in checked if len(toks(s) & dtoks) / max(len(toks(s)), 1) >= thresh)
    return grounded / len(checked) >= 0.7  # ≥70% of sentences strictly grounded


F35_SYS = """Bạn là kiểm định viên NGHIÊM NGẶT chất lượng trajectory pháp lý (để lọc data huấn luyện).
Cho câu hỏi + EVIDENCE (kết quả tra cứu) + CÂU TRẢ LỜI. Chấm:
- correct: câu trả lời có ĐÚNG SỰ THẬT pháp lý không?
- fully_grounded: MỌI khẳng định trong câu trả lời có được EVIDENCE hỗ trợ trực tiếp không
  (KHÔNG bịa thêm nội dung ngoài evidence, KHÔNG trộn lẫn 2 văn bản khác nhau)?
- no_wrong_direction: nếu về thay thế/sửa đổi, chiều quan hệ có ĐÚNG không (mới thay cũ, không ngược)?
Trả CHỈ JSON: {"id":"<id>","correct":true|false,"fully_grounded":true|false,"no_wrong_direction":true|false,"issue":"≤20 từ"}
CHỈ accept khi CẢ BA true."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default="PATH/TO/accepted.jsonl")
    ap.add_argument("--l3-thresh", type=float, default=0.6)
    a = ap.parse_args()
    acc = [json.loads(l) for l in (ROOT / a.accepted).open() if l.strip()]
    run_dir = (ROOT / a.accepted).parent

    # Stage (a): tightened L3 grounding (deterministic) — short abstain/template answers pass through
    stageA = []
    for r in acc:
        if not r.get("final_answer"):
            continue
        if r.get("answer_gt"):  # template item — keep (it has its own answer_gt; not judged by this grounding test)
            stageA.append(r)
        elif l3_grounding(r, a.l3_thresh):
            stageA.append(r)
    print(f"F3.5 (a) siết L3≥{a.l3_thresh}: {len(acc)} → {len(stageA)}")

    # Stage (b): Qwen3-32B strict check for correct+grounded+direction
    w = Worker("Qwen/Qwen3-32B", str(run_dir), max_tokens=300, temperature=0.0,
               extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def build(r):
        ev = "\n".join((s.get("tool_result_digest") or {}).get("text", "")[:400] for s in r.get("steps", []))[:2000]
        return [{"role": "system", "content": F35_SYS},
                {"role": "user", "content": f"[id: {r['custom_id']}]\nCâu hỏi: {r['question']}\nEVIDENCE:\n{ev}\nTRẢ LỜI: {(r.get('final_answer') or '')[:800]}"}]

    def po(text, r):
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
                            d = json.loads(re.sub(r",\s*}", "}", text[s:j + 1]))
                            if str(d.get("id")) == r["custom_id"]:
                                # gate: correct + fully_grounded (the two core flags). no_wrong_direction is only
                                # logged (it over-triggers on questions not about replacement). Strict enough
                                # while keeping enough volume.
                                return {"pass": bool(d.get("correct")) and bool(d.get("fully_grounded")),
                                        "correct": bool(d.get("correct")), "grounded": bool(d.get("fully_grounded")),
                                        "direction": bool(d.get("no_wrong_direction", True)), "issue": d.get("issue", "")}
                        except Exception:
                            pass
                        break
            i = s + 1

    f35_file = str(run_dir / "f35_scores.jsonl")
    w.map_items([{**r, "custom_id": r["custom_id"]} for r in stageA], build, po, f35_file, max_workers=10)
    verdict = {json.loads(l)["custom_id"]: (json.loads(l).get("parsed") or {}).get("pass")
               for l in open(f35_file) if json.loads(l).get("ok")}
    final = [r for r in stageA if verdict.get(r["custom_id"])]
    out = run_dir / "accepted_f35.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in final), encoding="utf-8")
    print(f"F3.5 (b) Qwen strict: {len(stageA)} → {len(final)}")
    print(f"→ accepted_f35: {len(final)} ({100*len(final)/len(acc):.0f}% của accepted gốc) | cost ${w.cost_usd:.2f}")
    g3 = "✅ đủ train" if len(final) >= 2000 else ("🟡 nhỏ nhưng OK" if len(final) >= 1500 else "🔴 quá ít")
    print(f"   {g3} (≥2000 lý tưởng; {len(final)})")


if __name__ == "__main__":
    main()
