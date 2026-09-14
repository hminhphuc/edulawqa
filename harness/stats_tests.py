#!/usr/bin/env python3
"""stats_tests.py — Project-standard statistical tests.

API (import from analysis scripts; do not copy the formulas elsewhere):
  wilcoxon_paired_1sided(deltas)         — /25 paired, alternative='greater'
  bootstrap_ci_median(deltas, alpha, n)  — one-/two-sided CI for the median delta
  rank_biserial(deltas)                  — effect size
  holm_correction(pvals)                 — Holm family of four primary comparisons (P1/P2/P4/P5); P3 was retired, see the paper's Statistics paragraph
  mcnemar(b, c)                          — MCQ: b = only system A correct, c = only system B correct
  noninferiority_mcq(acc_a, acc_b, n, delta_margin) — CI lower bound > −δ (not a p>0.05 test)

Unit-test: python3 harness/stats_tests.py (synthetic data with known answers)
"""
from __future__ import annotations
import random

from scipy.stats import wilcoxon, binomtest


def wilcoxon_paired_1sided(deltas: list[float]) -> float:
    nz = [d for d in deltas if d != 0]
    if len(nz) < 5:
        return 1.0
    return float(wilcoxon(nz, alternative="greater").pvalue)


def bootstrap_ci_median(deltas, alpha=0.10, n=10000, seed=20260704, two_sided=False):
    rng = random.Random(seed)
    meds = sorted(
        sorted(rng.choices(deltas, k=len(deltas)))[len(deltas) // 2]
        for _ in range(n))
    if two_sided:
        return meds[int(alpha / 2 * n)], meds[int((1 - alpha / 2) * n) - 1]
    return meds[int(alpha * n)]  # one-sided lower bound


def sign_ratio(deltas) -> float:
    """Sign ratio: (positive pairs − negative pairs)/nonzero pairs.

    NOT rank-biserial; kept because some internal tables used it.
    """
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    return (pos - neg) / max(pos + neg, 1)


def rank_biserial(deltas) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test.

    r = (W+ − W−) / (W+ + W−), where W+/W− are the rank sums of |delta| over the
    positive/negative pairs (pairs with delta = 0 are dropped, matching scipy's
    zero_method='wilcox'). This is the formula that produces the effect sizes
    reported in the paper (P1 0.35, P2 0.77).
    """
    from scipy.stats import rankdata
    nz = [d for d in deltas if d != 0]
    if not nz:
        return 0.0
    ranks = rankdata([abs(d) for d in nz])
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total else 0.0


def holm_correction(pvals: dict[str, float]) -> dict[str, tuple[float, bool]]:
    """→ {name: (p_adjusted, significant@0.05)} — Holm-Bonferroni."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max, rejected_stop = {}, 0.0, False
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        sig = (running_max <= 0.05) and not rejected_stop
        if not sig:
            rejected_stop = True
        out[name] = (round(running_max, 5), sig)
    return out


def mcnemar(b: int, c: int) -> float:
    """Exact McNemar (binomial). b,c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5).pvalue)


def noninferiority_mcq(correct_a: list[bool], correct_b: list[bool],
                       delta_margin: float = 0.02, n_boot=10000, seed=20260704):
    """A = distilled, B = base. Non-inferior if the CI95 lower bound of (accA−accB) > −margin."""
    assert len(correct_a) == len(correct_b)
    rng = random.Random(seed)
    n = len(correct_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(correct_a[i] for i in idx) / n
        db = sum(correct_b[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lower = diffs[int(0.025 * n_boot)]
    return {"delta_acc": round(sum(correct_a) / n - sum(correct_b) / n, 4),
            "ci95_lower": round(lower, 4), "margin": delta_margin,
            "non_inferior": lower > -delta_margin}


def _unittest():
    rng = random.Random(7)
    # 1. Wilcoxon: a clear +2 shift must give a small p; a zero shift a large p
    up = [2 + rng.gauss(0, 1) for _ in range(40)]
    flat = [rng.gauss(0, 1) for _ in range(40)]
    assert wilcoxon_paired_1sided(up) < 0.001, "shift+2 phải significant"
    assert wilcoxon_paired_1sided(flat) > 0.05, "shift 0 không được significant"
    # 2. bootstrap: median shift +2 → lower bound > 0
    assert bootstrap_ci_median(up) > 0
    assert bootstrap_ci_median(flat) <= 0.5
    # 3. rank-biserial has the correct sign
    assert rank_biserial(up) > 0.8 and abs(rank_biserial(flat)) < 0.4
    # 4. Holm: p=[.001,.01,.02,.04,.4] @m=5 → first three significant, last one not
    h = holm_correction({"a": .001, "b": .01, "c": .02, "d": .04, "e": .4})
    assert h["a"][1] and h["b"][1] and not h["e"][1]
    # 5. McNemar: 30 vs 5 discordant → very small p; 17 vs 15 → not significant
    assert mcnemar(30, 5) < 0.001 and mcnemar(17, 15) > 0.5
    # 6. Non-inferiority: equal accuracy at n=700 → non_inferior True; ~5 points worse → False
    a = [rng.random() < 0.7 for _ in range(700)]
    r = noninferiority_mcq(a, a, 0.02)
    assert r["non_inferior"], "identical phải non-inferior"
    b_worse = [x and rng.random() > 0.07 for x in a]
    r2 = noninferiority_mcq(b_worse, a, 0.02)
    assert not r2["non_inferior"], "kém ~5 điểm % phải FAIL margin 2%"
    print("stats_tests: 6/6 unit-test PASS")


if __name__ == "__main__":
    _unittest()
