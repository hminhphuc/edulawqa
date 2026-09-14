# Schema

All files are JSON Lines: one JSON object per line, UTF-8.

---

## `data/trajectories/edulawqa_trajectories.jsonl`

| Field | Type | Description |
|---|---|---|
| `messages` | array | Chat transcript. First turn `system`, then alternating `user` and `assistant`, ending on `assistant`. |
| `meta.qid` | string | Question identifier. 1,003 distinct values across 1,342 records. Prefixes: `q-V`, `q-G`, `q-T`, `q-H`, `q-R` for generated questions by route intent, `tmpl-` for deterministically templated ones. |
| `meta.route` | string | Route the teacher **declared** in its first assistant turn: `VECTOR`, `GRAPH`, or `HYBRID`. Not the tools it called — see data card 4.2. |

**Turn structure.** 5 to 11 turns, always odd: one system turn, then one user/assistant pair per tool step, then the final answer. Turn counts map to tool steps as 5→1, 7→2, 9→3, 11→4 (699, 364, 196 and 83 records respectively).

- `system` — identical across all records: the agent contract (emit a route decision, call tools, answer with citations).
- `user` (first) — the question, 11 to 92 words, median 24.
- `user` (subsequent) — a truncated tool-result *digest*, stored at the 1,200-character log cap. This is what the **students** saw during training. The **teacher** that produced the assistant turns saw the same digest up to 2,500 characters, so an answer may cite evidence that the stored digest truncates (data card 4.11). The digest is the evidence budget the paper analyses: the narrow configuration keeps 6 snippets of 350 characters.
- `assistant` (non-final) — route decision, structured thought fields (not raw chain-of-thought), and a tool call, as JSON.
- `assistant` (final) — a JSON object with `final_answer` (free text) and `citations` (a list of `{document, article}` pairs the model cites). Every final answer mentions at least one document number in its text; 6 records have an empty `citations` list while still naming a document in the prose. Final answers mention 1.57 document numbers on average; 981 distinct documents appear anywhere in the set (evidence digests included).

**Training use.** In the paper, user turns are loss-masked and assistant turns are trained.

Example, truncated:

```json
{
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý pháp lý giáo dục Việt Nam theo kiểu agent gọi công cụ. Với mỗi câu hỏi: (1) phát route_decision chọn nguồn, (2) gọi công cụ truy xuất, (3) trả lời…"},
    {"role": "user", "content": "Câu hỏi: Theo quy định, nếu chủ đầu tư không thông báo cho địa phương trước khi xây dựng công trình có yếu tố nước ngoài…"},
    {"role": "assistant", "content": "{\"route_decision\": {\"label\": \"VECTOR\", \"sub\": null, \"rationale\": \"…\"}, \"tool_call\": {…}}"}
  ],
  "meta": {"qid": "q-V-00000", "route": "VECTOR"}
}
```

---

## `data/benchmarks/edulawqa_route.jsonl` — EduLawQA-Route

| Field | Type | Description |
|---|---|---|
| `id` | string | Question identifier, `Q001`–`Q200`. |
| `question` | string | The question, in Vietnamese. |
| `category` | string | One of ten: `lookup` (24), `definition` (6), `tuition` (7), `condition` (22), `rights` (37), `tuyen_sinh` (9), `procedure` (18), `comparison` (5), `scenario` (32), `new_2025` (40). |
| `category_vi` | string | Vietnamese label for the same category. |
| `difficulty` | string | `easy` (60), `medium` (100), `hard` (40). |
| `route_reference` | string | Majority route label from three cross-family LLM annotators: `VECTOR` (163), `GRAPH` (17), `HYBRID` (20). A reference signal, not ground truth: pairwise kappa 0.27-0.53. |
| `route_unanimous` | bool | True for the 140 questions where all three annotators agreed. |
| `route_votes` | object | The three individual votes, keyed by annotator family, so disagreement is inspectable. |

**No reference answers.** Scoring uses the 25-point rubric judge in `harness/judge.py` under two frozen configurations, strict and trust-citation, which are never compared against each other. The route fields above are a reference signal, not ground truth (pairwise kappa 0.27–0.53, data card 4.5).

---

## `data/benchmarks/edulawqa_multisource.jsonl` — EduLawQA-MultiSource

| Field | Type | Description |
|---|---|---|
| `id` | string | Question identifier. |
| `question` | string | Deterministically template-generated, no LLM involved. |
| `template` | string | The generating template. |
| `gold_docs` | array | Exactly two document identifiers. Correct answers must draw on both. |
| `anchor` | string | The real amendment relation the pair is anchored in. 95 distinct documents across 120 questions. |
| `category` | string | Always `multi_source_amendment`. |
| `difficulty` | string | Difficulty label. |

## `data/benchmarks/edulawqa_multisource_nonum.jsonl`

Same fields, plus:

| Field | Type | Description |
|---|---|---|
| `orig_question` | string | The question before document numbers were stripped. |
| `variant` | string | Always `no_document_number`. |

97 of the 120 questions have a strippable document number; the rest are absent by construction. This is the control that separates a genuine conditional routing policy from surface pattern-matching on document numbers.
