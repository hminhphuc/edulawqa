# LexRoute — EduLawQA artifacts

[![validate](https://github.com/hminhphuc/edulawqa/actions/workflows/validate.yml/badge.svg)](https://github.com/hminhphuc/edulawqa/actions/workflows/validate.yml)
[![data licence: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE)
[![code licence: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

Data, prompts, and evaluation harness for ***Trajectory Distillation for Faithful Statutory Citations in Vietnamese Education Law QA*** (KSE 2026).

The paper distils agentic behaviour — tool use, grounding, citation discipline — from an open-weights teacher (Qwen3-32B) into 4B and 8B students, and asks what that supervision actually buys. This repository holds everything needed to inspect or reuse that work, short of the corpus itself.

---

## What is here

```
data/trajectories/edulawqa_trajectories.jsonl   1,342 citation-validated agentic trajectories
data/benchmarks/edulawqa_route.jsonl              200 questions, in-domain benchmark
data/benchmarks/edulawqa_multisource.jsonl        120 amendment-anchored multi-document questions
data/benchmarks/edulawqa_multisource_nonum.jsonl   97 same questions, document numbers removed
prompts/                                          the three prompts that drive filtering and judging
harness/                                          rubric judge, CVRS filters, statistics, agreement
scripts/                                          filter, router, citation and entailment scorers
configs/                                          the frozen scoring rubric and its checksum
results/                                          the result files the paper promises:
                                                  the filtering audit's raw judgements at
                                                  both accepted tiers, their metrics, and
                                                  the per-track external table (both modes)
docs/DATA_CARD.md                                 what the data is, how it was built, what is wrong with it
docs/SCHEMA.md                                    field-by-field description
```

Install dependencies with `pip install -r requirements.txt`. The rubric judge and the
entailment scorer each call an external API and read their configuration from the
environment (`GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` for the Vertex AI
judge; the entailment scorer's API key); everything else runs offline.

**Read [`docs/DATA_CARD.md`](docs/DATA_CARD.md) before using the data.** Sections 4 and 5 document every known characteristic and limitation, including the ones that do not flatter us.

## Verify before you trust

Every factual claim in the data card is re-derived from the files by one script:

```bash
python3 scripts/validate_release.py            # hard checks + documented characteristics
python3 scripts/validate_release.py --verbose  # adds corpus reach and citation statistics
```

It exits non-zero if any hard check fails, and it runs on every push: the badge above
is this script passing on the files in this repository. Checksums are in `data/SHA256SUMS`:

```bash
cd data && sha256sum -c SHA256SUMS
```

The scoring rubric is frozen. `harness/judge.py` verifies the rubric file against the
checksum in `configs/FROZEN_CHECKSUMS.md` on every load and refuses to run on a mismatch,
so a silently edited rubric cannot reach a reported number.

## Three things worth knowing immediately

1. **1,342 trajectories cover 1,003 questions.** 339 questions carry two trajectories from k=2 sampling. They are not duplicates — highest pairwise Jaccard is 0.907, under the 0.92 deduplication threshold. Deduplicate on `meta.qid` if you want one per question.
2. **`meta.route` is the route the teacher *declared*, not the tools it *called*.** HYBRID is 1.4% of declared labels but 16.2% of enacted behaviour. Section 4.2 of the data card gives the full cross-tabulation. State which one you mean.
3. **This is not legal advice.** Questions are synthetic, answers came from a language model, and no lawyer certified them. Citation metrics score existence, lexical support, and entailment — none certifies that a cited provision *governs* the question.

## Answers to the two reviewer questions

One KSE 2026 reviewer asked for two details the six-page limit could not hold. Both answers live here in full:

- **The F3.5 threshold and its re-verification prompt** — [`prompts/f35_reverification.md`](prompts/f35_reverification.md). Short version: 0.4 → 0.6 is one threshold being tightened, not a range, and the test changes scope from a window around each citation to every sentence of the answer. The value was fixed once after an audit and never tuned.
- **The entailment judge's configuration and calibration** — [`prompts/nli_citation_judge.md`](prompts/nli_citation_judge.md). Includes the verbatim prompt, the scale (576 + 635 citations, 100% identifier match), and a plain statement that it was never calibrated against professional legal annotation.

The route codebook behind the silver labels and the benchmark labels is [`prompts/route_codebook_v2.md`](prompts/route_codebook_v2.md).

## What is not here

- **The corpus.** Vietnamese education-law documents are public and carry no copyright, but the indexed corpus is large and infrastructure-specific. Trajectories quote the retrieved excerpts they used, so the evidence a model saw is visible without it.
- **Model weights.** The students are QLoRA adapters over public base models; the recipe is in the paper.
- **Run artifacts.** Per-run metrics live in the research repository. The ones the paper
  explicitly promises are here in `results/`: the filtering audit's raw judgements at both
  accepted tiers (paper: "raw released") and the per-track external table carrying the
  conservative scores (paper: "conservative scores in the released artifacts").

## Reproducing the evaluation

The harness expects a retrieval backend; see the paper's method section for the tool contract. With your own retriever wired in:

```bash
python3 harness/judge.py       --help   # 25-point rubric, strict and trust-citation configurations
python3 scripts/citation_accuracy.py --help   # judge-free citation existence and support
python3 scripts/e37_nli_citation.py  --help   # entailment cross-check (needs an API key in the environment)
python3 harness/stats_tests.py --help   # paired Wilcoxon, exact McNemar, TOST, bootstrap
```

No script contains a hard-coded key or an absolute path; all read from the environment and resolve paths relative to the repository, overridable with `LEXROUTE_ROOT`.

## Licence

Data and prompts under **CC BY 4.0**. Code under **MIT**. Trajectories were produced by Qwen3-32B under Apache-2.0, which permits redistributing outputs. Quoted statutory text comes from Vietnamese legal normative documents, which carry no copyright under Article 15 of the Vietnamese Intellectual Property Law. Full statement in data card section 8.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contact

Issues and corrections through the repository issue tracker. Corresponding author details are in the paper.
