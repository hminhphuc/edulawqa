#!/usr/bin/env python3
"""e04_worker.py — synchronous multi-provider worker for open-weights teacher inference.

Used to generate teacher trajectories, questions, and silver route labels. Design notes:
  - OpenAI-compatible client: DeepInfra primary, Together fallback (separate cost multiplier)
  - retry with backoff by error code; 500-record batch files; resume by custom_id
  - real cost tracking from usage tokens, written to <run_dir>/monitor.jsonl
  - custom_id embedded in the prompt and echoed back in JSON output, so results are
    re-aligned by identifier rather than by line order

Contains no task logic: the caller supplies build_messages(item) and parse_output(text).
API keys are read from the environment (DEEPINFRA_API_KEY, TOGETHER_API_KEY); none is
stored in this file.
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROVIDERS = {
    "deepinfra": {"base": "https://api.deepinfra.com/v1/openai/chat/completions",
                   "key_env": "DEEPINFRA_API_KEY", "cost_multiplier": 1.0},
    "together": {"base": "https://api.together.xyz/v1/chat/completions",
                  "key_env": "TOGETHER_API_KEY", "cost_multiplier": 3.5},
}
# $/1M tokens (in, out) at the primary provider
PRICE = {"Qwen/Qwen3-235B-A22B-Instruct-2507": (0.13, 0.60),
         "Qwen/Qwen3-32B": (0.10, 0.30),
         "Qwen/Qwen3-30B-A3B": (0.08, 0.29)}


class Worker:
    def __init__(self, model: str, run_dir: str, max_tokens: int = 6000,
                 temperature: float = 0.7, cost_cap_usd: float | None = None,
                 extra_body: dict | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}  # e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        self.cost_cap = cost_cap_usd
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = self.run_dir / "monitor.jsonl"
        self.cost_usd = 0.0
        self.n_done = 0
        self.n_fail = 0
        # resume the cost counter from an existing monitor log
        if self.monitor.exists():
            for l in self.monitor.open():
                try:
                    self.cost_usd = json.loads(l).get("cost_usd", self.cost_usd)
                except Exception:
                    pass

    def _log(self, **kw):
        rec = {"ts": time.strftime("%H:%M:%S"), "n_done": self.n_done, "n_fail": self.n_fail,
               "cost_usd": round(self.cost_usd, 4), **kw}
        with self.monitor.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def chat(self, messages: list[dict], provider: str = "deepinfra", retries: int = 4) -> dict:
        """→ {text, tokens_in, tokens_out, provider} — fails over to the next provider on hard errors."""
        order = [provider] + [p for p in PROVIDERS if p != provider]
        last = ""
        for prov in order:
            cfg = PROVIDERS[prov]
            key = os.environ.get(cfg["key_env"], "")
            if not key:
                continue
            for att in range(retries):
                if self.cost_cap and self.cost_usd >= self.cost_cap:
                    raise RuntimeError(f"COST-KILL: {self.cost_usd:.2f} >= cap {self.cost_cap} (cost cap)")
                try:
                    r = requests.post(cfg["base"],
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": self.model, "messages": messages,
                              "max_tokens": self.max_tokens, "temperature": self.temperature,
                              **self.extra_body},
                        timeout=300)
                    if r.status_code == 429:
                        time.sleep(20 * (att + 1))
                        continue
                    r.raise_for_status()
                    d = r.json()
                    u = d.get("usage", {})
                    ti, to = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                    pin, pout = PRICE.get(self.model, (0.15, 0.6))
                    self.cost_usd += cfg["cost_multiplier"] * (ti * pin + to * pout) / 1e6
                    return {"text": d["choices"][0]["message"]["content"] or "",
                            "tokens_in": ti, "tokens_out": to, "provider": prov}
                except Exception as e:
                    last = str(e)[:150]
                    time.sleep(5 * (att + 1))
            self._log(event="provider_failover", from_provider=prov, err=last)
        raise RuntimeError(f"mọi provider fail: {last}")

    def map_items(self, items: list[dict], build_messages, parse_output,
                  out_file: str, max_workers: int = 8, log_every: int = 25):
        """Run in parallel, resume by custom_id, append one record per item.
        parse_output(text, item) → dict or None (None = fail; the raw text is stored for diagnosis)."""
        outp = Path(out_file)
        done = set()
        if outp.exists():
            done = {json.loads(l)["custom_id"] for l in outp.open() if l.strip()}
            print(f"[worker] resume: {len(done)} đã có")
        todo = [it for it in items if it["custom_id"] not in done]
        assert len({it["custom_id"] for it in todo}) == len(todo), "custom_id trùng (guardrail 1)"
        lock_buf = []

        def work(it):
            try:
                r = self.chat(build_messages(it))
                parsed = parse_output(r["text"], it)
                self.n_done += 1
                rec = {"custom_id": it["custom_id"],
                       "ok": parsed is not None,
                       "parsed": parsed,
                       "raw_text": None if parsed is not None else r["text"][:2000],
                       "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                       "provider": r["provider"]}
                if parsed is None:
                    self.n_fail += 1
                return rec
            except RuntimeError as e:
                self.n_fail += 1
                return {"custom_id": it["custom_id"], "ok": False, "error": str(e)[:200]}

        with ThreadPoolExecutor(max_workers=max_workers) as ex, outp.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(ex.map(work, todo), 1):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % log_every == 0:
                    f.flush()
                    self._log(event="progress", total=len(todo), done_now=i,
                              fail_rate=round(self.n_fail / max(self.n_done + self.n_fail, 1), 3))
        self._log(event="finished", total=len(todo))
        print(f"[worker] xong {len(todo)} | fail {self.n_fail} | cost ${self.cost_usd:.2f}")


if __name__ == "__main__":
    # smoke test: 2 questions through Qwen3-32B (cheap) — checks the client and the cost log
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    w = Worker("Qwen/Qwen3-32B", "runs/_worker_smoke", max_tokens=200, temperature=0)
    items = [{"custom_id": f"smoke-{i}", "q": q} for i, q in
             enumerate(["Điều 1 Luật Giáo dục 2019 nói về gì? Trả lời 1 câu.",
                        "Viên chức là gì? Trả lời 1 câu."])]
    w.map_items(items,
                build_messages=lambda it: [{"role": "user", "content": it["q"]}],
                parse_output=lambda t, it: {"answer": t.strip()[:200]},
                out_file="runs/_worker_smoke/out.jsonl", max_workers=2)
    for l in open("runs/_worker_smoke/out.jsonl"):
        r = json.loads(l)
        print(f"  {r['custom_id']}: ok={r['ok']} | {str(r.get('parsed',{}).get('answer'))[:80]}")
