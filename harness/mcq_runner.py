#!/usr/bin/env python3
"""mcq_runner.py — Project-standard MCQ runner.

Built-in policies:
  - custom_id per question; the model is asked to echo the id — primary alignment is still
    structural (one call per question), the echo is only a cross-check.
  - Strip <think>...</think> BEFORE extracting the answer (otherwise thinking arms are penalised unfairly).
  - Multi-stage answer extraction: JSON {"answer":"A"} → an "Answer: X" phrase (Vietnamese or
    English) → a single letter on the first/last line.
  - Per-arm parse-fail rate is logged; above 1% on any arm → BLOCK reporting that arm's numbers
    (a red warning is printed).
  - Permutation-invariance check: n questions are run under 2 different option permutations —
    does the model pick by CONTENT (correct) or by POSITION (position bias)? The rate is reported.
answer_fn: callable(prompt_text) -> raw model text, where prompt_text is built by build_prompt(item). The runner is model-agnostic
(local vLLM or an API can be plugged in) — fairness is kept at the arm-config layer.
"""
from __future__ import annotations
import json
import random
import re
import unicodedata

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
LETTERS = ["A", "B", "C", "D"]


def strip_think(text: str) -> str:
    out = THINK_RE.sub("", text or "")
    # unclosed <think> tag (truncated output) → drop everything from <think> onward
    if "<think>" in out:
        out = out.split("<think>")[0]
    return out.strip()


def extract_answer(raw: str) -> str | None:
    """→ 'A'|'B'|'C'|'D' or None (parse-fail)."""
    t = strip_think(raw)
    if not t:
        return None
    # 1. JSON {"answer": "A"}
    m = re.search(r'"answer"\s*:\s*"?([A-Da-d])"?', t)
    if m:
        return m.group(1).upper()
    # 2. "Answer: X" / "Choose X" phrasing, Vietnamese or English (see the regex)
    m = re.search(r"(?:đáp án|answer|chọn|trả lời)\s*(?:là|:)?\s*\**([A-Da-d])\b", t, re.I)
    if m:
        return m.group(1).upper()
    # 3. first/last line consisting of a single letter
    for line in [t.splitlines()[0], t.splitlines()[-1]]:
        m = re.fullmatch(r"\**([A-Da-d])[\.\)]?\**", line.strip())
        if m:
            return m.group(1).upper()
    # 4. first standalone letter of the form "A." / "(A)"
    m = re.search(r"(?:^|\s)\(?([A-D])[\.\)]", t)
    if m:
        return m.group(1).upper()
    return None


def build_prompt(item: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return (f"[id: {item['custom_id']}]\nCâu hỏi trắc nghiệm pháp luật. Chọn MỘT đáp án đúng.\n\n"
            f"{item['question']}\n\n{opts}\n\n"
            f'Trả về CHỈ JSON: {{"id": "{item["custom_id"]}", "answer": "<A|B|C|D>"}}')


def run_mcq(items: list[dict], answer_fn, arm_name: str = "?") -> dict:
    """items: [{custom_id, question, options{A..D}, correct_answer}] → results + guard."""
    ids = [it["custom_id"] for it in items]
    assert len(set(ids)) == len(ids), "duplicate custom_id: results could not be re-aligned by identifier"
    rows, fails = [], 0
    for it in items:
        raw = answer_fn(build_prompt(it))
        pred = extract_answer(raw)
        if pred is None:
            fails += 1
        rows.append({"custom_id": it["custom_id"], "pred": pred,
                     "correct": it["correct_answer"].strip().upper(),
                     "is_correct": pred == it["correct_answer"].strip().upper(),
                     "raw_head": strip_think(raw)[:80]})
    n = len(rows)
    acc = sum(r["is_correct"] for r in rows) / max(n - fails, 1)
    fail_rate = fails / max(n, 1)
    ok_to_use = fail_rate <= 0.01
    if not ok_to_use:
        print(f"🔴 [{arm_name}] parse-fail {fail_rate:.1%} > 1% — BLOCKING: do not report numbers for this arm")
    return {"arm": arm_name, "n": n, "acc_on_parsed": round(acc, 4),
            "parse_fail_rate": round(fail_rate, 4), "ok_to_use": ok_to_use, "rows": rows}


def permute_item(item: dict, seed: str) -> dict:
    rng = random.Random(seed)
    perm = LETTERS[:]
    rng.shuffle(perm)
    return {**item,
            "options": {LETTERS[i]: item["options"][perm[i]] for i in range(4)},
            "correct_answer": LETTERS[perm.index(item["correct_answer"].strip().upper())]}


def permutation_check(items: list[dict], answer_fn, n_check: int = 50) -> dict:
    """Run n questions under 2 permutations — measures consistency BY CONTENT (not by position)."""
    rng = random.Random(20260704)
    sample = rng.sample(items, min(n_check, len(items)))
    agree = same_position = 0
    for it in sample:
        p1 = permute_item(it, f"perm1-{it['custom_id']}")
        p2 = permute_item(it, f"perm2-{it['custom_id']}")
        a1 = extract_answer(answer_fn(build_prompt(p1)))
        a2 = extract_answer(answer_fn(build_prompt(p2)))
        if a1 and a2:
            c1 = p1["options"].get(a1)
            c2 = p2["options"].get(a2)
            if c1 == c2:
                agree += 1          # same CONTENT → good invariance
            if a1 == a2:
                same_position += 1  # same POSITION → sign of position bias when high while agree is low
    return {"n": len(sample), "content_consistent": agree,
            "position_same": same_position,
            "verdict": "OK" if agree >= 0.8 * len(sample) else "⚠ position-bias khả nghi"}


def _unittest():
    # extract_answer across formats
    assert extract_answer('{"id":"x","answer":"C"}') == "C"
    assert extract_answer("<think>lan man rất dài...</think>Đáp án: B") == "B"
    assert extract_answer("Sau khi phân tích, tôi chọn A.") == "A"
    assert extract_answer("D") == "D"
    assert extract_answer("<think>chưa xong bị cắt") is None
    assert extract_answer("Không thể xác định") is None
    # strip_think keeps the trailing part
    assert strip_think("<think>x</think>OK") == "OK"
    # permute preserves the content of the correct answer
    it = {"custom_id": "t1", "question": "?", "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
          "correct_answer": "C"}
    p = permute_item(it, "s")
    assert p["options"][p["correct_answer"]] == "3", "permute phải bảo toàn nội dung đáp án đúng"
    # run_mcq with a fake answer_fn: the model always answers by CONTENT '3'
    def fake_fn(prompt):
        for line in prompt.splitlines():
            m = re.match(r"([A-D])\. 3$", line.strip())
            if m:
                return f'{{"answer":"{m.group(1)}"}}'
        return "hỏng"
    r = run_mcq([{**permute_item(it, f"s{i}"), "custom_id": f"t{i}"} for i in range(10)], fake_fn, "fake")
    assert r["acc_on_parsed"] == 1.0 and r["parse_fail_rate"] == 0
    pc = permutation_check([it] * 10, fake_fn, 10)
    assert pc["content_consistent"] == pc["n"], "content-consistent model phải đạt 100%"
    print("mcq_runner: 9/9 unit-test PASS")


if __name__ == "__main__":
    _unittest()
