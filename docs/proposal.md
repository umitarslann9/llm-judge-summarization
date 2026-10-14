# CENG467 — Topic Proposal

**Proposed Project Title:** Large vs. Small LLMs as Evaluators of Abstractive Summarization

| Field | Value |
|---|---|
| Proposal ID | P-09 |
| Group ID | 24 |
| Student | Ümit Arslan (290201003) |
| Contact Email | umtarsln0@gmail.com |
| Instructor Status | Approved |

## Problem Description

The LLM-as-a-Judge approach is widely used to evaluate text generation because large models like GPT-4 or Claude capture semantic and factual quality better than metrics such as ROUGE or BERTScore. However, large judges are expensive and closed-source, which makes them impractical for large-scale or iterative evaluation. It is still unclear whether a small open-source LLM can serve as a reliable judge instead. This project asks a simple question for abstractive summarization: does using a small judge actually hurt evaluation quality, and if so, by how much?

## Proposed Approach

The project implements a single LLM-as-a-Judge pipeline and runs it with two different judges: a large proprietary model (GPT-4 or Claude, accessed via API) and a small open-source model (Qwen2.5, run locally). Both judges receive the same prompt and score each summary on four criteria — faithfulness, coherence, relevance, and fluency — using a short Chain-of-Thought format where the model reasons briefly before giving a numerical score. The two judges are then compared on how well each aligns with human scores from SummEval, how often they agree with each other, and how much they differ in cost and speed. The goal is to determine whether the small judge is accurate enough to replace the large one in practice.

## Baseline Methods

1. **ROUGE (ROUGE-1, ROUGE-2, ROUGE-L):** Standard n-gram overlap against human reference summaries. Serves as the classical lexical baseline.
2. **BERTScore:** Embedding-based semantic similarity against human references. A stronger semantic baseline than ROUGE and a common default in modern summarization evaluation.
3. **Zero-shot Direct Scoring (Qwen2.5, no rubric, no CoT):** The same small model used as the main small judge, but prompted to output a single numerical score without any rubric or reasoning step. This shows how much of the small judge's performance comes from prompt design rather than raw model ability.

## Dataset (Name & Source)

SummEval (Fabbri et al., 2021) — https://huggingface.co/datasets/mteb/summeval. (Not all, to be decided later)

## Evaluation Strategy & Metrics

Human alignment is measured using Spearman and Kendall-τ correlation between each judge's scores and SummEval human scores, reported per dimension and averaged. Agreement between the two judges is measured using Pearson correlation and the mean score difference across all summaries. Cost and speed are also reported in terms of API tokens, GPU-hours, and average latency per summary, and plotted against human correlation to visualize the cost–quality trade-off. The small judge is considered a viable replacement if its human correlation stays close to that of the large judge while running at significantly lower cost. Both judges are additionally compared against ROUGE and BERTScore to confirm that LLM-based evaluation improves over classical metrics.

## Expected Challenges

Small models sometimes produce malformed outputs, which are handled with structured JSON formatting and a regex fallback. The two judges may also use the 1–5 scale with different strictness, so both raw and standardized scores are reported to make the comparison fair. API budget for the large judge is limited, so a subset of SummEval is evaluated rather than the full dataset. Because the subset is limited in size, bootstrapping is used to estimate confidence intervals on the correlation metrics.
