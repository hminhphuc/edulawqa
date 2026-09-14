# External evaluation, per track (both scoring modes)

This file accompanies the external-benchmark table of the paper. Every number is regenerated from the run logs by script in the research repository; nothing in this file is typed by hand.

Two kinds of change between versions are labelled. **[measurement fix]**: the harness truncation cap that cut retrieved evidence (600 to 6,000 characters) was removed and *every* arm was re-run under the same setting; v1 numbers are kept for transparency. **[adapter]**: forced-choice decoding, in which items the student abstained on are committed by a second pass that never alters an already answered item; the conservative (abstention-preserving) score is reported next to it. No model weights changed and the frozen in-domain benchmark was untouched.

## ALQAC (statute-grounded multiple choice / true-false, exact match) [measurement fix]

| arm | ALQAC'25 v1 | ALQAC'25 v2 conservative | ALQAC'25 v2 forced | ALQAC'23 v1 | ALQAC'23 v2 conservative | ALQAC'23 v2 forced |
|---|---|---|---|---|---|---|
| 8B-traj | 66.2% | 76.1% | 78.9% | 77.8% | 81.1% | 81.1% |
| 4B-traj | 64.8% | 74.6% | 77.5% | 71.1% | 76.7% | 78.9% |
| 8B-answer | 76.1% | 76.1% | 76.1% | 80.0% | 80.0% | 80.0% |
| 4B-answer | 71.8% | 71.8% | 71.8% | 70.0% | 70.0% | 70.0% |

ALQAC'25 has 71 items and ALQAC'23 has 90. The education-law items among them (2 and 10 respectively) are part of this table; the paper's leakage check excludes them and reports the shift. The ALQAC'25 gold contains disputed items: on author inspection against the statute text, 7 of 14 8B-trajectory errors examined had a gold answer that conflicts with the cited provision; they are counted as errors here.

## VLSP 2025 Legal-SLM shared task, official public test (exact match) [adapter]

| task / arm | conservative | forced | accuracy on parsed items | organizer reference |
|---|---|---|---|---|
| NLI 8B-traj | 78.7% | 88.0% | 91.5% | organizer baseline Qwen3-4B-Base 0.97; top system 0.98 |
| NLI 4B-traj | 31.3% | 57.3% | 90.4% |  |
| NLI 8B-answer | 86.7% | 86.7% | 86.7% |  |
| NLI 4B-answer | 42.0% | 80.0% | 78.8% |  |
| QA 8B-traj | 57.5% | 78.8% | 79.2% | organizer baseline Qwen3-4B-Base 0.821; top system 0.927 |
| QA 4B-traj | 13.0% | 82.2% | 90.5% |  |
| QA 8B-answer | 82.9% | 82.9% | 82.9% |  |
| QA 4B-answer | 81.5% | 81.5% | 81.5% |  |

Conservative scores are low for the trajectory students because they abstain when the retrieved context does not settle the item; on the items they do answer (accuracy on parsed items) the trajectory student leads the answer-only student in three of four cells, VLSP-QA at 8B being the exception.

## ViHERMES multi-hop (healthcare regulation; GPT-4o-judged correctness) [measurement fix]

| arm | accuracy v2 | hop 1 | hop 2 | hop 3 | hop 4 | hop 5 |
|---|---|---|---|---|---|---|
| 8B-traj | 47.3% | 93% | 57% | 47% | 27% | 13% |
| 8B-answer | 41.3% | 93% | 43% | 30% | 23% | 17% |
| 4B-traj | 43.3% | 93% | 40% | 40% | 30% | 13% |
| 4B-answer | 40.7% | 93% | 50% | 20% | 20% | 20% |

Versus the truncated-context v1 run, the 8B trajectory student moves hop 2: 37% to 57%, hop 3: 33% to 47%, hop 4: 10% to 27%; most of the apparent multi-hop decay was a context-truncation artifact. Hop-5 accuracy stays low for every arm and is a real limit of sub-9B students.

## ViRHE4QA (university training regulations; token-F1 and ROUGE-L with gold context)

| arm | token-F1 | ROUGE-L |
|---|---|---|
| 8B-traj | 0.6524 | 0.6113 |
| 8B-answer | 0.6392 | 0.5971 |
| 4B-traj | 0.7393 | 0.6822 |
| 4B-answer | 0.7334 | 0.6860 |

Zero-shot, education-law-distilled students given the gold context. No in-domain baseline is listed: the figure previously used for comparison (0.689) could not be traced to the public version of the R2GQA paper.

## Reading the table
- The truncation cap was a measurement defect (it cut the deciding provision); removing it and re-running every arm is a correction, not tuning.
- Forced choice is an adapter. Conservative scores (calibrated abstention, the behaviour a legal assistant should have) and forced scores (comparable to leaderboards) are both reported; the paper's table shows the forced mode.
- Remaining limits: disputed ALQAC'25 gold, hop-5 questions on ViHERMES, statutes enacted after the students' knowledge cutoff, and sub-9B reasoning.
