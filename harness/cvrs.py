#!/usr/bin/env python3
"""cvrs.py — CVRS F2: hard citation validation for trajectories.

F2 has three layers; its thresholds are hard and are never relaxed:
  L1 WHITELIST_SKY: every document number cited in final_answer/citations MUST appear in
     the evidence (the tool-step digests). A citation outside the whitelist is hallucinated.
  L2 INTERVAL-OVERLAP: each cited (document, article) pair must match one of the
     "[document-number <article> d]" markers in a digest (see DIGEST_PAIR_RE).
  L3 HALLUGRAPH-LITE: the context around each citation in the answer must have token
     overlap ≥0.4 with the digest of the cited document/article (guards against
     "right citation, different claim").
Valid exception: an ABSTAIN answer ("information not found"-style) needs no citation and
passes with category "abstain" (intended for refuse-style questions; on an ordinary question
an abstention still passes F2 but scores low at F3).

API: f2_check(traj_record) → {"f2_pass": bool, "category": "cited|abstain|uncited",
     "flags": [...], "metrics": {...}}
Unit-test: python3 harness/cvrs.py
"""
from __future__ import annotations
import json
import re
import unicodedata

# NOTE: the character class MUST include 0-9 — document numbers such as "43/2019/QH14" carry digits in the suffix
SKY_RE = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ][A-ZĐa-z0-9\-\.]*\b")
DIGEST_PAIR_RE = re.compile(r"\[(\d{1,4}/\d{4}/[A-ZĐ][A-ZĐa-z0-9\-\.]*)\s+Điều\s+(\w+)\]")
ABSTAIN_PAT = re.compile(r"không\s+tìm\s+thấy\s+thông\s+tin|không\s+có\s+(đủ\s+)?thông\s+tin|"
                         r"không\s+thể\s+trả\s+lời|chưa\s+đủ\s+(căn\s+cứ|thông\s+tin)")


def _norm(s):
    return unicodedata.normalize("NFC", s or "")


def _tokens(s, min_len=3):
    s = re.sub(r"\s+", " ", _norm(s).casefold())
    return {t for t in re.findall(r"[\wà-ỹ]+", s) if len(t) >= min_len}


def _evidence(traj) -> tuple[set, set, dict]:
    """→ (whitelist of document numbers, set of (document, article-str) pairs, digest text keyed by document number)."""
    wl, pairs, dig = set(), set(), {}
    for s in traj.get("steps", []):
        d = (s.get("tool_result_digest") or {}).get("text", "") or ""
        for m in SKY_RE.finditer(d):
            sky = re.sub(r"\s+", "", m.group(0))
            wl.add(sky)
            dig[sky] = dig.get(sky, "") + "\n" + d
        for m in DIGEST_PAIR_RE.finditer(d):
            pairs.add((m.group(1), str(m.group(2))))
    return wl, pairs, dig


def f2_check(traj: dict) -> dict:
    ans = _norm(traj.get("final_answer") or "")
    flags, metrics = [], {}
    wl, pairs, dig = _evidence(traj)

    cits = traj.get("citations") or []
    cited = []
    for c in cits:
        sky = re.sub(r"\s+", "", _norm(str(c.get("so_ky_hieu") or "")))
        if sky:
            cited.append((sky, str(c.get("dieu") or "").strip() or None))
    # document numbers cited inline in the answer text also count (guards against citations that bypass the citations field)
    ans_skys = {re.sub(r"\s+", "", m.group(0)) for m in SKY_RE.finditer(ans)}

    is_abstain = bool(ABSTAIN_PAT.search(ans[:200].casefold()))
    if not cited and not ans_skys:
        if is_abstain:
            return {"f2_pass": True, "category": "abstain", "flags": [], "metrics": {"n_cited": 0}}
        return {"f2_pass": False, "category": "uncited",
                "flags": ["answer_khong_citation"], "metrics": {"n_cited": 0}}

    # L1 whitelist
    hallucinated = [s for s in {s for s, _ in cited} | ans_skys if s not in wl]
    if hallucinated:
        flags.append(f"L1_sky_ngoai_whitelist:{hallucinated[:3]}")
    # L2 interval (sky, dieu)
    n_pair = n_pair_ok = 0
    for sky, dieu in cited:
        if dieu is None:
            continue
        n_pair += 1
        if (sky, dieu) in pairs or sky in wl and not pairs:
            n_pair_ok += 1
        elif (sky, dieu) not in pairs:
            flags.append(f"L2_pair_khong_khop:{sky}Đ{dieu}")
    metrics["interval_score"] = round(n_pair_ok / n_pair, 3) if n_pair else 1.0
    # L3 hallugraph-lite: ±120-character context around each inline citation of a document number in the answer
    overlaps = []
    for sky in {s for s, _ in cited} | ans_skys:
        if sky not in dig:
            continue
        for m in re.finditer(re.escape(sky), ans):
            ctx = ans[max(0, m.start() - 120):m.end() + 120]
            ov = len(_tokens(ctx) & _tokens(dig[sky])) / max(len(_tokens(ctx)), 1)
            overlaps.append(ov)
    metrics["grounding_score"] = round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0
    weak = [o for o in overlaps if o < 0.4]
    if weak:
        flags.append(f"L3_grounding_yeu:{len(weak)}/{len(overlaps)}")
    metrics.update({"n_cited": len(cited), "n_hallucinated_sky": len(hallucinated)})
    l2_fail = any(f.startswith("L2") for f in flags)
    return {"f2_pass": not hallucinated and not l2_fail and not weak,
            "category": "cited", "flags": flags, "metrics": metrics}


def _unittest():
    good = {"steps": [{"tool_result_digest": {"text": "[43/2019/QH14 Điều 114] Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2020. Luật Giáo dục số 38/2005/QH11 hết hiệu lực."}}],
            "citations": [{"so_ky_hieu": "43/2019/QH14", "dieu": "114"}],
            "final_answer": "Luật Giáo dục 2019 có hiệu lực thi hành từ ngày 01 tháng 7 năm 2020 [43/2019/QH14 Điều 114]."}
    r = f2_check(good)
    assert r["f2_pass"], f"trajectory chuẩn phải pass: {r}"
    # L1: cited document number absent from the evidence
    bad1 = {**good, "citations": [{"so_ky_hieu": "99/2099/NĐ-CP", "dieu": "1"}],
            "final_answer": "Theo [99/2099/NĐ-CP Điều 1] thì abc."}
    r1 = f2_check(bad1)
    assert not r1["f2_pass"] and any("L1" in f for f in r1["flags"])
    # L2: right document, wrong article
    bad2 = {**good, "citations": [{"so_ky_hieu": "43/2019/QH14", "dieu": "5"}],
            "final_answer": "Điều 5 quy định X [43/2019/QH14 Điều 5]."}
    r2 = f2_check(bad2)
    assert not r2["f2_pass"] and any("L2" in f for f in r2["flags"])
    # L3: correct citation but the claim is unrelated to the digest
    bad3 = {**good, "final_answer": "Mức phạt tiền tối đa với hành vi dạy thêm sai quy định là năm mươi triệu đồng theo [43/2019/QH14 Điều 114]."}
    r3 = f2_check(bad3)
    assert not r3["f2_pass"] and any("L3" in f for f in r3["flags"]), r3
    # abstention without citations → passes as "abstain"
    ab = {"steps": [], "citations": [], "final_answer": "Không tìm thấy thông tin trong nguồn được cung cấp."}
    assert f2_check(ab)["category"] == "abstain" and f2_check(ab)["f2_pass"]
    # ordinary answer without citations → fail
    un = {"steps": [], "citations": [], "final_answer": "Mức học phí là 500 nghìn."}
    assert not f2_check(un)["f2_pass"] and f2_check(un)["category"] == "uncited"
    print("cvrs F2: 6/6 unit-test PASS")


if __name__ == "__main__":
    _unittest()
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        print("Usage: python3 harness/cvrs.py <trajectories.jsonl>")
        raise SystemExit(0)
    if len(sys.argv) > 1:  # real run on a trajectory file
        rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
        rows = [r for r in rows if not r.get("error")]
        res = [f2_check(r) for r in rows]
        n_pass = sum(1 for x in res if x["f2_pass"])
        from collections import Counter
        print(f"{sys.argv[1]}: F2 pass {n_pass}/{len(res)} ({100*n_pass/max(len(res),1):.0f}%) | "
              f"category {dict(Counter(x['category'] for x in res))}")
        flag_c = Counter(f.split(':')[0] for x in res for f in x['flags'])
        print(f"  flags: {dict(flag_c)}")
