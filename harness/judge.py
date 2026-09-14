#!/usr/bin/env python3
"""harness/judge.py — The project's single judge wrapper.

Every /25 score MUST go through this module. Source of truth: configs/judge.frozen.yaml
(SHA verified on every load; a mismatch refuses to run).

Built-in policies:
  - custom_id: required per item; the judge echoes `id` in its JSON and the wrapper asserts the match.
  - parse-fail/echo-fail → up to 2 retries with a format reminder → null.
  - consensus: median over ≥2 successful runs; fewer → consensus_total=None (the caller drops the item PAIRWISE).
  - parse_fail_rate is logged; the caller is responsible for warning above 2% per arm.

The user-turn prompt template is reproduced verbatim from the original calibrated judge.
Changing the template changes the measuring instrument and must be recorded as a
MEASUREMENT event.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("LEXROUTE_ROOT", Path(__file__).resolve().parent.parent))
# Repo root (this file lives in harness/). Override with LEXROUTE_ROOT if relocated.
FROZEN_YAML = ROOT / "configs" / "judge.frozen.yaml"
EVAL_FREEZE = ROOT / "configs" / "FROZEN_CHECKSUMS.md"

CRITERIA = ["factual_accuracy", "citation_quality", "reasoning_quality",
            "completeness", "language_quality"]
FORMAT_REMINDER = ("\n\nNHẮC LẠI: Trả về CHỈ MỘT JSON object đúng schema đã nêu "
                   "(không markdown, không ```), và trường \"id\" phải là id đã cho.")
ABSTAIN_PAT = re.compile(
    r"không\s+tìm\s+thấy\s+thông\s+tin|không\s+có\s+(đủ\s+)?thông\s+tin|"
    r"không\s+thể\s+trả\s+lời|chưa\s+đủ\s+(căn\s+cứ|thông\s+tin)")


def load_frozen_config() -> dict:
    """Load the frozen rubric and verify its SHA-256 against configs/FROZEN_CHECKSUMS.md.

    A mismatch is a hard stop: it means the measuring instrument changed after freezing.
    """
    raw = FROZEN_YAML.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    recorded = re.search(r"`([0-9a-f]{64})`", EVAL_FREEZE.read_text(encoding="utf-8"))
    if not recorded or recorded.group(1) != sha:
        raise RuntimeError(
            f"judge.frozen.yaml SHA={sha[:16]}... does not match configs/FROZEN_CHECKSUMS.md — "
            "the scoring rubric was modified after freezing. Refusing to score.")
    cfg = yaml.safe_load(raw.decode("utf-8"))
    cfg["_sha256"] = sha
    return cfg


def build_prompt(item: dict) -> str:
    """Verbatim mirror of the original calibrated judge's build_prompt."""
    q = item.get("q") or item.get("question", "")
    answer = item.get("answer", "") or ""
    cat = item.get("category", "")
    diff = item.get("difficulty", "")
    cits = item.get("citations", []) or []
    cit_block = ""
    if cits:
        cit_lines = []
        for c in cits:
            sky = c.get("so_ky_hieu", "")
            dieu = c.get("dieu", "")
            khoan = c.get("khoan", "")
            nd = (c.get("noi_dung", "") or "")[:200]
            cit_lines.append(f"  - {sky} Đ{dieu}/K{khoan}: {nd}")
        cit_block = "\n**Citations kèm theo:**\n" + "\n".join(cit_lines[:10])

    return f"""## Câu hỏi đánh giá [{item.get('id','?')}]

**Câu hỏi:** {q}
**Loại:** {cat} | **Mức độ:** {diff}

**Câu trả lời:**
---
{answer}
---
{cit_block}

Đánh giá theo 5 tiêu chí (thang 1-5). Score guidance:

**1. factual_accuracy** — số liệu/ngày/điều khoản/tên văn bản đúng?
  5 = mọi sự thật được cite đều khớp với citation provided
  4 = phần lớn đúng, có 1-2 chi tiết phụ chưa rõ
  3 = nhìn chung đúng nhưng có 1-2 chỗ chưa chính xác
  2 = nhiều chi tiết sai
  1 = sai cơ bản hoặc bịa số/ngày không tồn tại

**2. citation_quality** — citations cụ thể (số ký hiệu + điều/khoản)?
  5 = mọi câu khẳng định đều được cite chính xác từ chunks
  4 = cite đầy đủ nhưng có 1 chỗ thiếu/lờ mờ
  3 = có cite nhưng không đủ rõ
  2 = cite mơ hồ hoặc không đủ
  1 = không cite hoặc cite document không tồn tại

**3. reasoning_quality** — lập luận chặt chẽ?
  5 = logic đầy đủ, kết luận đúng từ căn cứ
  4 = logic tốt, có vài chỗ cần triển khai hơn
  3 = logic chấp nhận được
  2 = lập luận lỏng lẻo
  1 = không có lập luận / sai logic

**4. completeness** — bao phủ ý chính của câu hỏi?
  5 = đầy đủ ý chính + bổ sung context phù hợp
  4 = đầy đủ ý chính
  3 = đáp ứng cơ bản
  2 = thiếu một số ý quan trọng
  1 = không trả lời được câu hỏi
  * Nếu honest "Không tìm thấy thông tin", thường ở mức 3 (acceptable abstention)

**5. language_quality** — tiếng Việt pháp lý chuẩn, rõ?
  5 = chuẩn pháp lý, súc tích, dễ hiểu
  4 = tốt, có thể tinh chỉnh
  3 = tạm ổn
  2 = lỗi diễn đạt nhỏ
  1 = khó hiểu / sai văn phong

Trả về CHỈ JSON sau (KHÔNG markdown, KHÔNG ```):
{{
  "id": "{item.get('id','?')}",
  "scores": {{"factual_accuracy": <1-5>, "citation_quality": <1-5>, "reasoning_quality": <1-5>, "completeness": <1-5>, "language_quality": <1-5>}},
  "total": <tổng>,
  "explanations": {{"factual_accuracy": "<≤30 từ>", "citation_quality": "<≤30 từ>", "reasoning_quality": "<≤30 từ>", "completeness": "<≤30 từ>", "language_quality": "<≤30 từ>"}},
  "errors_found": [],
  "strengths": "<điểm mạnh nếu có>",
  "overall_verdict": "<nhận xét 1-2 câu>"
}}"""


def parse_json(text: str) -> dict | None:
    """Mirror of the original judge's parse_json — tolerant of markdown fences and trailing commas."""
    def try_parse(s: str):
        s = re.sub(r'"total"\s*:\s*(\d+)/\d+', r'"total": \1', s)
        s = re.sub(r",\s*([}\]])", r"\1", s.strip())
        try:
            d = json.loads(s)
            if "scores" in d:
                return d
        except Exception:
            pass
        return None
    i = 0
    while True:
        s = text.find("{", i)
        if s == -1:
            break
        depth = 0
        for j in range(s, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    r = try_parse(text[s:j + 1])
                    if r:
                        return r
                    break
        i = s + 1
    return None


class FrozenJudge:
    def __init__(self):
        self.cfg = load_frozen_config()
        # Credentials and project come from the environment only:
        #   GOOGLE_APPLICATION_CREDENTIALS  path to a service-account JSON
        #   GOOGLE_CLOUD_PROJECT            the Vertex AI project id
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        # --help and offline inspection must work without credentials; the check below
        # only fires when a judge is actually constructed to score something.
        if not project:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS) "
                               "to run the frozen judge against Vertex AI.")
        from google import genai
        m = self.cfg["model"]
        self.model = m["judge_model"]
        self.max_tokens = int(m.get("max_output_tokens", 4096))
        self.client = genai.Client(vertexai=True,
                                   project=project,
                                   location=m["location"])
        self.prompts = {
            "trust_citation": self.cfg["system_prompt_trust_citation"],
            "strict": self.cfg["system_prompt_strict"],
        }
        self.consensus_temps = self.cfg["protocol"]["consensus"]["temperatures"]
        self.single_temp = self.cfg["protocol"]["single_pass"]["temperature"]

    def _gen_config(self, config_name: str, temperature: float):
        from google.genai.types import GenerateContentConfig, ThinkingConfig
        kw = {"system_instruction": self.prompts[config_name],
              "temperature": temperature, "max_output_tokens": self.max_tokens}
        try:
            kw["thinking_config"] = ThinkingConfig(thinking_level="minimal")
            return GenerateContentConfig(**kw)
        except Exception:
            kw.pop("thinking_config", None)
            return GenerateContentConfig(**kw)

    def _call_once(self, item: dict, config_name: str, temperature: float) -> dict | None:
        """One scoring pass; on parse-fail/echo-fail, retry up to 2 times with a format reminder."""
        sid = str(item.get("id", ""))
        prompt = build_prompt(item)
        for attempt in range(3):  # first attempt + 2 retries
            p = prompt if attempt == 0 else prompt + FORMAT_REMINDER
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=p,
                    config=self._gen_config(config_name, temperature))
                parsed = parse_json(r.text or "")
            except Exception as e:
                msg = str(e)
                wait = 30 * (attempt + 1) if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 5
                time.sleep(wait)
                parsed = None
            if parsed and str(parsed.get("id", "")) == sid:
                t = parsed.get("total", 0)
                if isinstance(t, str):
                    m = re.match(r"[\d.]+", t.strip())
                    t = float(m.group()) if m else 0
                # total must equal the sum of the scores (guards against judge arithmetic errors)
                ssum = sum(float(parsed.get("scores", {}).get(k, 0) or 0) for k in CRITERIA)
                parsed["total"] = ssum if abs(float(t) - ssum) > 0.01 else float(t)
                # Deterministic ABSTAIN-CAP (enforces frozen rubric principle E: an abstention
                # scores 14-18/25; the model judge ignores this instruction, so it is enforced in code)
                ans_head = (item.get("answer") or "")[:150].casefold()
                if ABSTAIN_PAT.search(ans_head):
                    parsed["is_abstain"] = True
                    if parsed["total"] > 18:
                        parsed["abstain_capped_from"] = parsed["total"]
                        parsed["total"] = 18.0
                parsed["_temp"] = temperature
                return parsed
        return None  # parse/echo fail after 3 attempts

    def judge_items(self, items: list[dict], config_name: str,
                    mode: str = "consensus", max_workers: int = 8) -> list[dict]:
        """Score a list of items. Every item MUST have a unique 'id' (custom_id).

        mode='consensus' → 3 runs at the frozen temperatures, median (primary arms)
        mode='single'    → 1 run at 0.1 (exploratory arms)
        """
        ids = [str(it.get("id", "")) for it in items]
        assert len(ids) == len(set(ids)) and all(ids), "custom_id must be unique and non-empty: results are re-aligned by identifier"
        temps = self.consensus_temps if mode == "consensus" else [self.single_temp]

        def work(item):
            runs = [self._call_once(item, config_name, t) for t in temps]
            ok = [r for r in runs if r]
            if mode == "consensus":
                consensus = statistics.median([r["total"] for r in ok]) if len(ok) >= 2 else None
            else:
                consensus = ok[0]["total"] if ok else None
            return {"custom_id": str(item["id"]), "config": config_name, "mode": mode,
                    "runs": runs, "n_run_ok": len(ok), "consensus_total": consensus,
                    "parse_fail": consensus is None}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            out = list(ex.map(work, items))
        n_fail = sum(1 for o in out if o["parse_fail"])
        if n_fail:
            print(f"[judge] CẢNH BÁO: {n_fail}/{len(out)} item parse-fail sau retry "
                  f"({100*n_fail/len(out):.1f}%) — dropped PAIRWISE in paired comparisons")
        return out


if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("Environment: GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS are")
        print("required to score; the rubric is verified against configs/FROZEN_CHECKSUMS.md")
        print("on every load and a mismatch refuses to run.")
        raise SystemExit(0)
    j = FrozenJudge()
    print(f"Frozen judge OK: {j.model} | SHA {j.cfg['_sha256'][:16]}… | "
          f"consensus temps {j.consensus_temps}")
