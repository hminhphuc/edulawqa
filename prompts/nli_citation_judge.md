# NLI citation judge (entailment cross-check)

Reviewer question at KSE 2026: *"the exact configuration, prompt schema, and calibration for this entailment judge are largely omitted."*

## Why it exists

The judge-free metric `support@0.4` is lexical overlap, and it shares its threshold with training filter F2. A reviewer could reasonably object that it rewards extractiveness rather than grounding. This entailment check, in the spirit of ALCE, tests whether the ordering survives a different construct.

## Configuration

| Setting | Value |
|---|---|
| Model | `gpt-5-mini` (proprietary — used **only** as an evaluation judge, never on training data) |
| Decoding | zero-shot, provider defaults, `max_completion_tokens` 2000 |
| Unit of judgement | one citation (not one answer) |
| Inputs per call | the cited article's full text, the question, the system's answer |
| Output | strict JSON with an echoed `custom_id` |
| Scale | 576 student citations + 635 anchor citations, scored in two runs |
| Identifier match | 100%, 0 malformed records |
| Retries | up to 3 per request; failures written to `bad_records.jsonl`, never silently dropped |

The `custom_id` echo is a project-wide requirement: every batched LLM call carries a unique per-request identifier that the model must echo, and results are re-aligned by identifier rather than by line order.

## Reported metric

The NLI column in the paper counts **SUPPORTED + PARTIAL**. The stricter SUPPORTED-only rate is also reported in the paper's discussion (29.1% for the 8B trajectory student).

## Calibration — stated plainly

This judge was **not** calibrated against professional legal annotation. The only human reference in the study is a blind expert spot-check over 80 both-answered questions, whose citation-correctness rate (51.7%) sits between the lenient lexical construct and the strict entailment-only construct. Treat the NLI column as a second automated construct that happens to agree on ordering, not as ground truth.

## Prompt, verbatim (Vietnamese, as used)

```
Bạn là chuyên gia thẩm định trích dẫn pháp lý. Cho MỘT điều luật và MỘT câu trả lời của hệ thống QA, hãy đánh giá: nội dung pháp lý chính của CÂU TRẢ LỜI có được ĐIỀU LUẬT này chống đỡ (entail/support) không?

- SUPPORTED: các khẳng định pháp lý chính của câu trả lời được điều luật chống đỡ trực tiếp (kể cả khi diễn đạt khác — paraphrase vẫn tính).
- PARTIAL: điều luật chống đỡ một phần nội dung, phần còn lại không có trong điều luật này.
- NOT_SUPPORTED: điều luật không liên quan hoặc mâu thuẫn với nội dung trả lời.

ĐIỀU LUẬT ({doc} — Điều {dieu}):
{article}

CÂU HỎI: {question}

CÂU TRẢ LỜI CẦN THẨM ĐỊNH:
{answer}

Trả về JSON duy nhất, không thêm text:
{{"custom_id": "{cid}", "verdict": "SUPPORTED|PARTIAL|NOT_SUPPORTED"}}
```

English gloss: *"You are an expert reviewer of legal citations. Given ONE statutory article and ONE answer from a QA system, judge: are the main legal assertions of the ANSWER supported (entailed) by this article? **SUPPORTED** — the main legal assertions are directly supported (paraphrase counts). **PARTIAL** — the article supports part of the content; the rest is not in this article. **NOT_SUPPORTED** — the article is irrelevant to, or contradicts, the answer. Return a single JSON object, no extra text."*

Full implementation: `scripts/e37_nli_citation.py`.
