#!/usr/bin/env python3
"""validate_release.py — independent quality audit of the released LexRoute artifacts.

Everything claimed in docs/DATA_CARD.md sections 4 and 5 is re-derived here from the
files themselves. Run it before trusting the data, and run it again after any edit.

    python3 scripts/validate_release.py            # summary
    python3 scripts/validate_release.py --verbose  # per-check detail

Exit code 0 if every hard check passes, 1 otherwise. Soft findings (documented
characteristics, not defects) are printed but do not fail the run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "data/trajectories/edulawqa_trajectories.jsonl"
ROUTE = ROOT / "data/benchmarks/edulawqa_route.jsonl"
MULTI = ROOT / "data/benchmarks/edulawqa_multisource.jsonl"
NONUM = ROOT / "data/benchmarks/edulawqa_multisource_nonum.jsonl"

SKY = re.compile(r"\d{1,4}/\d{4}/[A-ZĐ][A-ZĐa-z0-9\-]*")
TOOLS = ("retrieve_vector", "retrieve_graph", "direct_fetch")

failures: list[str] = []
notes: list[str] = []


def load(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            failures.append(f"{path.name}:{i} is not valid JSON ({e})")
    return rows


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip().casefold()


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label + (f" — {detail}" if detail else ""))


def note(label: str, detail: str) -> None:
    print(f"  [note] {label} — {detail}")
    notes.append(f"{label}: {detail}")


def question_of(row: dict) -> str:
    for key in ("question", "q", "text"):
        if key in row:
            return row[key]
    return json.dumps(row, ensure_ascii=False)


def first_user_turn(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def assistant_text(row: dict) -> str:
    return " ".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant")


def tokens(s: str) -> set:
    return set(re.findall(r"\w+", unicodedata.normalize("NFC", s).casefold()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    for p in (TRAJ, ROUTE, MULTI, NONUM):
        if not p.exists():
            print(f"missing file: {p}", file=sys.stderr)
            return 1

    traj, route, multi, nonum = load(TRAJ), load(ROUTE), load(MULTI), load(NONUM)

    print("\n== 1. Record counts ==")
    check("trajectories = 1,342", len(traj) == 1342, f"found {len(traj)}")
    check("EduLawQA-Route = 200", len(route) == 200, f"found {len(route)}")
    check("EduLawQA-MultiSource = 120", len(multi) == 120, f"found {len(multi)}")
    check("MultiSource no-number = 97", len(nonum) == 97, f"found {len(nonum)}")

    print("\n== 2. Structural integrity ==")
    bad = []
    for i, r in enumerate(traj, 1):
        ms = r.get("messages")
        if not isinstance(ms, list) or len(ms) < 3:
            bad.append((i, "fewer than 3 turns"))
            continue
        roles = [m.get("role") for m in ms]
        if roles[0] != "system":
            bad.append((i, "first turn is not system"))
        if roles[-1] != "assistant":
            bad.append((i, f"last turn is {roles[-1]}"))
        if any(m.get("content") is None for m in ms):
            bad.append((i, "null content"))
        if not isinstance(r.get("meta", {}).get("qid"), str):
            bad.append((i, "missing meta.qid"))
    check("every trajectory well-formed", not bad, f"{len(bad)} malformed" + (f" e.g. {bad[:3]}" if bad else ""))
    check("single shared system prompt", len({r["messages"][0]["content"] for r in traj}) == 1)
    check("every final answer names ≥1 document number in its text",
          all(SKY.search(r["messages"][-1]["content"]) for r in traj))
    empty_struct = 0
    for r in traj:
        try:
            d = json.loads(r["messages"][-1]["content"])
        except (ValueError, TypeError):
            d = None
        if isinstance(d, dict) and "citations" in d and not d["citations"]:
            empty_struct += 1
    note("empty structured citation lists", f"{empty_struct} records name a document in prose but carry "
                                           f"citations=[] (schema note; data card section 6)")
    for name, rows, required in (("EduLawQA-Route", route, ("id", "question", "category", "difficulty")),
                                 ("MultiSource", multi, ("id", "question", "gold_docs", "anchor")),
                                 ("MultiSource no-number", nonum, ("id", "question", "gold_docs", "variant"))):
        miss = [r.get("id") for r in rows if any(k not in r for k in required)]
        check(f"{name} has all required fields", not miss, f"{len(miss)} incomplete")
    check("MultiSource has exactly 2 gold documents per question",
          all(len(r["gold_docs"]) == 2 for r in multi))

    print("\n== 3. Duplication ==")
    digests = collections.Counter(
        hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest() for r in traj)
    check("no byte-identical duplicate records", sum(c - 1 for c in digests.values() if c > 1) == 0)

    qids = [r["meta"]["qid"] for r in traj]
    per_qid = collections.Counter(qids)
    uniq = len(per_qid)
    check("1,003 distinct question ids", uniq == 1003, f"found {uniq}")
    note("k=2 sampling", f"{sum(1 for v in per_qid.values() if v == 2)} questions have 2 trajectories, "
                         f"{sum(1 for v in per_qid.values() if v == 1)} have 1 — by design, see data card 4.1")

    by_text = collections.defaultdict(set)
    for r in traj:
        by_text[norm(first_user_turn(r))].add(r["meta"]["qid"])
    collisions = {t: q for t, q in by_text.items() if len(q) > 1}
    check("no two question ids share identical text", not collisions, f"{len(collisions)} collisions")

    pairs = []
    grouped = collections.defaultdict(list)
    for r in traj:
        grouped[r["meta"]["qid"]].append(r)
    for rows in grouped.values():
        if len(rows) == 2:
            a, b = tokens(assistant_text(rows[0])), tokens(assistant_text(rows[1]))
            if a | b:
                pairs.append(len(a & b) / len(a | b))
    if pairs:
        check("no same-question pair exceeds the 0.92 dedup threshold", max(pairs) <= 0.92,
              f"max {max(pairs):.3f}")
        note("same-question similarity", f"median {statistics.median(pairs):.3f}, max {max(pairs):.3f} over {len(pairs)} pairs")

    for name, rows in (("EduLawQA-Route", route), ("MultiSource", multi), ("MultiSource no-number", nonum)):
        seen = {norm(question_of(r)) for r in rows}
        check(f"{name} has no internal duplicate questions", len(seen) == len(rows),
              f"{len(rows) - len(seen)} duplicates")

    print("\n== 4. Train/eval leakage (exact match) ==")
    train_q = {norm(first_user_turn(r)) for r in traj}
    for name, rows in (("EduLawQA-Route", route), ("MultiSource", multi), ("MultiSource no-number", nonum)):
        overlap = train_q & {norm(question_of(r)) for r in rows}
        check(f"training set disjoint from {name}", not overlap, f"{len(overlap)} shared questions")
    ms_docs = {d.strip().upper() for r in multi for d in r["gold_docs"]}
    train_docs = set()
    for r in traj:
        for m in r["messages"]:
            for d in SKY.findall(m.get("content", "")):
                train_docs.add(d.strip().upper())
    shared = ms_docs & train_docs
    note("document-level overlap", f"{len(shared)}/{len(ms_docs)} ({100*len(shared)/len(ms_docs):.0f}%) of "
                                    f"MultiSource gold documents also appear in training trajectories. Questions "
                                    f"do not overlap, but the documents do — both draw on the same education-law "
                                    f"corpus. MultiSource recall figures are therefore an upper bound; the paper "
                                    f"says so in its limitations.")

    routes_per_qid = collections.defaultdict(set)
    for r in traj:
        routes_per_qid[r["meta"]["qid"]].add(r["meta"].get("route"))
    disagree = sum(1 for v in routes_per_qid.values() if len(v) > 1)
    note("same-question route disagreement", f"{disagree} questions have two trajectories declaring "
                                             f"different meta.route values — data card 4.1")
    spaced = [i for i, r in enumerate(traj, 1)
              if re.search(r"\d{1,4}/\d{4}/ [A-ZĐ]|\d{1,4}/ \d{4}/|\d{1,4} /\d{4}/", first_user_turn(r))]
    if spaced:
        note("document numbers with stray spaces", f"{len(spaced)} records (lines {spaced}) — data card 4.8")
    EDU = re.compile(r"BGDĐT|BGDDT|giáo dục|học sinh|sinh viên|giáo viên|trường|nhà giáo|tuyển sinh|học phí|đào tạo", re.I)
    off = sum(1 for r in traj if not EDU.search(first_user_turn(r))
              and not EDU.search(r["messages"][-1]["content"]))
    note("domain heuristic", f"{off}/{len(traj)} ({100*off/len(traj):.0f}%) trajectories carry no education "
                             f"keyword in question or answer — corpus is education-law-centred, not -only "
                             f"(data card 4.10)")
    corrupt = sum(json.dumps(r, ensure_ascii=False).count("Phật tiền") for r in traj)
    note("source-crawl defect", f"'Phật tiền' (corrupted 'Phạt tiền') occurs {corrupt} times in quoted "
                                f"statutory text — data card 4.12")

    note("scope of this check", "exact text only; it cannot detect paraphrase. "
                                "The paper reports an embedding audit (median NN-cosine 0.73, one disclosed "
                                "near-duplicate at 0.955). Re-run an embedding check before training on this data.")

    print("\n== 5. Encoding and hygiene ==")
    nfd = [i for i, r in enumerate(traj, 1)
           if (s := json.dumps(r, ensure_ascii=False)) != unicodedata.normalize("NFC", s)]
    if nfd:
        note("NFD characters", f"{len(nfd)} records not NFC-normalised (lines {nfd}) — shipped as trained, "
                               f"see data card 4.6; run scripts/normalize_nfc.py if you need NFC")
    empty_num = [i for i, r in enumerate(traj, 1)
                 if re.search(r"(Thông tư|Nghị định|Quyết định|Luật)\s+/\d{4}/", first_user_turn(r))]
    if empty_num:
        note("empty document numbers", f"{len(empty_num)} template questions rendered an empty document "
                                       f"number (lines {empty_num}) — data card 4.8")

    ctrl = sum(1 for r in traj if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", json.dumps(r, ensure_ascii=False)))
    check("no control characters", ctrl == 0, f"{ctrl} records")

    secrets = {
        "Google API key": r"AIza[0-9A-Za-z_\-]{30,}",
        "OpenAI key": r"sk-[A-Za-z0-9]{32,}",
        "service-account key": r'"private_key"\s*:',
        "bearer token": r"Bearer\s+[A-Za-z0-9._\-]{24,}",
        "home path": r"/home/[a-z]+/",
    }
    for path in (TRAJ, ROUTE, MULTI, NONUM, *sorted((ROOT / "harness").glob("*.py")),
                 *sorted((ROOT / "scripts").glob("*.py"))):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = {k: len(re.findall(v, text)) for k, v in secrets.items()}
        hits = {k: v for k, v in hits.items() if v}
        if hits:
            check(f"no secrets in {path.name}", False, str(hits))
    check("no secrets or absolute home paths anywhere", not any(f.startswith("no secrets in") for f in failures))

    emails = set()
    for path in (TRAJ, ROUTE, MULTI, NONUM):
        emails |= set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                                 path.read_text(encoding="utf-8")))
    allowed = {"thongtinchinhphu@chinhphu.vn"}
    check("no unexpected email addresses", emails <= allowed, f"unexpected: {sorted(emails - allowed)}")
    if emails & allowed:
        note("known email", "thongtinchinhphu@chinhphu.vn is the published Government Portal address "
                            "inside a document signature block, not personal data (data card 4.7)")

    print("\n== 6. Label distributions (reported, not judged) ==")
    declared = collections.Counter(r["meta"].get("route") for r in traj)
    enacted = collections.Counter()
    for r in traj:
        used = {t for t in TOOLS if any(t in m.get("content", "")
                                        for m in r["messages"] if m.get("role") == "assistant")}
        if {"retrieve_vector", "retrieve_graph"} <= used:
            enacted["HYBRID"] += 1
        elif "retrieve_graph" in used:
            enacted["GRAPH"] += 1
        elif "retrieve_vector" in used:
            enacted["VECTOR"] += 1
        elif used:
            enacted["fetch-only"] += 1
        else:
            enacted["none"] += 1
    note("declared routes (meta.route)", str(dict(declared)))
    note("enacted routes (tools actually called)", str(dict(enacted)))
    note("why they differ", f"HYBRID is {100*declared.get('HYBRID',0)/len(traj):.1f}% of declared labels but "
                            f"{100*enacted.get('HYBRID',0)/len(traj):.1f}% of enacted behaviour — data card 4.2")
    note("EduLawQA-Route categories", str(dict(collections.Counter(r["category"] for r in route))))
    note("EduLawQA-Route difficulty", str(dict(collections.Counter(r["difficulty"] for r in route))))

    if args.verbose:
        print("\n== 7. Category x difficulty cross-tabulation (EduLawQA-Route) ==")
        cats = sorted({r["category"] for r in route})
        diffs = ["easy", "medium", "hard"]
        cell = collections.Counter((r["category"], r["difficulty"]) for r in route)
        print(f"  {'category':<12} " + " ".join(f"{d:>7}" for d in diffs))
        for c in cats:
            print(f"  {c:<12} " + " ".join(f"{cell.get((c, d), 0):>7}" for d in diffs))
        empty = sum(1 for c in cats for d in diffs if (c, d) not in cell)
        print(f"  {empty} of {len(cats)*len(diffs)} cells empty — category and difficulty are partly "
              f"confounded (data card 4.9)")

        print("\n== 8. Corpus reach ==")
        docs = collections.Counter()
        for r in traj:
            for m in r["messages"]:
                for d in SKY.findall(m.get("content", "")):
                    docs[d] += 1
        print(f"  {len(docs)} distinct legal documents referenced; most frequent: {docs.most_common(5)}")
        cites = [len(set(SKY.findall(r['messages'][-1]['content']))) for r in traj]
        print(f"  citations per final answer: mean {sum(cites)/len(cites):.2f}, distribution {dict(collections.Counter(cites))}")

    print("\n" + "=" * 64)
    if failures:
        print(f"FAILED — {len(failures)} hard check(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ALL HARD CHECKS PASSED ({len(notes)} documented characteristics noted above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
