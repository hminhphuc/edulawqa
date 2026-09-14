# Changelog

## v1.0 — September 2026

First public release, matching the KSE 2026 camera-ready paper.

- 1,342 citation-validated agentic trajectories over Vietnamese education law, covering
  1,003 questions (339 carry a second trajectory from k=2 sampling).
- Three benchmarks: EduLawQA-Route (200 questions with route labels), EduLawQA-MultiSource
  (120 amendment-anchored multi-document questions) and its 97-question no-number control.
- Evaluation harness: rubric judge against a frozen rubric, CVRS citation filters, the F3.5
  re-verification filter, paired statistics, and inter-annotator agreement.
- The three prompts that drive filtering, entailment judging and route annotation.
- Result files the paper promises: the filtering audit's raw judgements at both accepted
  tiers with their metrics, and the per-track external results in both scoring modes.
- `scripts/validate_release.py` re-derives every factual claim in the data card from the
  files and exits non-zero if a hard check fails.

Known characteristics are documented in Section 4 of the data card rather than silently
fixed, so that anyone reusing the data can decide what matters for their purpose.
