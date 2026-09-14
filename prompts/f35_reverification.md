# F3.5 — semantic tier (teacher re-verification)

Reviewer question at KSE 2026: *"why a threshold range (0.4-0.6) is used instead of a fixed mathematical cutoff, nor does it provide the specific prompt used for this critical teacher re-verification step."*

Answer: **0.4 → 0.6 is not a range.** It is one threshold being tightened, and the test itself changes scope at the same time.

## Stage (a) — deterministic grounding test

Tier F2 measures token overlap inside a ±120-character window around each citation, and flags the trajectory if overlap falls below **0.4**. F3.5 replaces that with a stricter test applied to **every sentence of the answer** at **0.6**:

- a sentence of ≥ 6 words is grounded if ≥ 60% of its content tokens (length ≥ 3) occur in the retrieved digests;
- a trajectory passes if ≥ 70% of its qualifying sentences are grounded.

Source, verbatim:

```python
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
```

The 0.6 value was **fixed once** after an audit found CVRS-accepted trajectories only 40–45% correct, and was never swept or tuned. Two independent runs of the filter (`f35_run.log`, `f35_run2.log`) both used 0.6 and both produced 4,864 → 2,626 → 2,283 → 1,342.

## Stage (b) — teacher re-verification prompt

Model: Qwen3-32B (the same open-weights teacher that generated the trajectories), temperature 0, max 300 tokens. A trajectory is accepted only when **both** `correct` and `fully_grounded` are true; `no_wrong_direction` is logged but does not gate.

System prompt, verbatim (Vietnamese, as used):

```
Bạn là kiểm định viên NGHIÊM NGẶT chất lượng trajectory pháp lý (để lọc data huấn luyện).
Cho câu hỏi + EVIDENCE (kết quả tra cứu) + CÂU TRẢ LỜI. Chấm:
- correct: câu trả lời có ĐÚNG SỰ THẬT pháp lý không?
- fully_grounded: MỌI khẳng định trong câu trả lời có được EVIDENCE hỗ trợ trực tiếp không
  (KHÔNG bịa thêm nội dung ngoài evidence, KHÔNG trộn lẫn 2 văn bản khác nhau)?
- no_wrong_direction: nếu về thay thế/sửa đổi, chiều quan hệ có ĐÚNG không (mới thay cũ, không ngược)?
Trả CHỈ JSON: {"id":"<id>","correct":true|false,"fully_grounded":true|false,"no_wrong_direction":true|false,"issue":"≤20 từ"}
CHỈ accept khi CẢ BA true.
```

English gloss: *"You are a STRICT inspector of legal trajectory quality (for filtering training data). Given a question + EVIDENCE (retrieval results) + ANSWER, score: **correct** — is the answer legally true? **fully_grounded** — is EVERY assertion in the answer directly supported by the EVIDENCE (nothing invented beyond the evidence, no mixing of two different documents)? **no_wrong_direction** — for amendment/replacement relations, is the direction right (new replaces old, not the reverse)? Return ONLY JSON: {"id": "...", "correct": true|false, "fully_grounded": true|false, "no_wrong_direction": true|false, "issue": "≤20 words"}. Accept ONLY when all three are true."*

The prompt's closing line tells the model to accept only when all three flags are true; the pipeline deliberately gates on the first two (`correct` and `fully_grounded`) and only logs `no_wrong_direction` (`scripts/f35_filter.py`, accept rule). The paper reports the implemented gate, not the prompt's wording.

Full implementation: `scripts/f35_filter.py`.
