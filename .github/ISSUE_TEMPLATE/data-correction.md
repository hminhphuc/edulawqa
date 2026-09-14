---
name: Data correction
about: Report an incorrect answer, citation, or label in the data
title: "[data] "
labels: data
---

**File and line**
Which file, and which line number or `id` / `meta.qid`.

**What is wrong**
Quote the field as it stands.

**What it should be, and why**
Where possible, cite the provision or the source document.

**Have you run the validator?**
`python3 scripts/validate_release.py` — say whether it flags this case. Several known
characteristics are documented rather than fixed; Section 4 of the data card lists them.
