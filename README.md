# Large vs. Small LLMs as Evaluators of Abstractive Summarization

**CENG 467 – Natural Language Understanding and Generation – Term Project**
Ümit Arslan (290201003) · Group 24 · Proposal ID P-09

---

## 1. What this project does

This project tests whether a **small, open LLM** can replace a **large LLM**
as an automatic *judge* for abstractive summarization quality.

The "LLM-as-a-Judge" setup is widely used because strong models align with human
quality judgments far better than ROUGE or BERTScore – but the strongest judges are
large, costly, and often closed. The core question:

> **Does swapping a large judge for a small open judge meaningfully hurt evaluation quality, and by how much?**

This is an **evaluation task**, not a generation task. The system does **not** write
summaries. It scores summaries that already exist in the SummEval benchmark and
measures how well those scores correlate with human ratings.

**Input:** `(source document, machine-generated summary, scoring rubric)`
**Output:** four numeric scores – *faithfulness, coherence, relevance, fluency* –
each preceded by a short Chain-of-Thought (CoT) justification.

---

## 2. Components

| Component | Role |
|---|---|
| **Baseline 1 – ROUGE** (ROUGE-1/2/L) | Classical n-gram overlap vs. reference summaries |
| **Baseline 2 – BERTScore** | Embedding-based semantic similarity vs. reference summaries |
| **Baseline 3 – Zero-shot Direct Scoring** | Small model, single number, *no rubric, no CoT* – isolates prompt-design effect |
| **Small Judge** | Qwen2.5-3B (local, via Ollama) + full rubric + CoT |
| **Large Judge** | Gemini 2.5 Flash (proprietary, via the Google Gemini API) + full rubric + CoT – reference judge |

All five run on the **same fixed SummEval subset** so comparisons are apples-to-apples.

> **Note on judge models.** Following the proposal's intent (a small local judge vs.
> a large proprietary one), the small judge is Qwen2.5-3B run locally via Ollama and
> the large judge is Google Gemini 2.5 Flash via the Gemini API (pay-as-you-go; a
> full 800-record run costs well under US$1). The code also ships a **Groq** backend
> (`provider: groq`, e.g. Llama-3.3-70B) as a fully free alternative large judge.

---

## 3. Repository structure

```
CENG-467-Term-Project/
├── common.py             # shared config/.env loading, JSONL IO, paths, dim mapping
├── data/
│   ├── download.py       # pulls mteb/summeval from the HF Hub, prints schema
│   ├── preprocess.py     # builds the fixed, seeded subset -> subset.jsonl
│   └── subset.jsonl      # generated – the frozen evaluation set (do not edit by hand)
├── prompts/
│   ├── rubric_cot.txt    # 4-dimension rubric + CoT instruction (used by both judges)
│   └── direct_score.txt  # minimal "score this summary" prompt (Baseline 3)
├── baselines/
│   ├── rouge_baseline.py
│   ├── bertscore_baseline.py
│   └── zeroshot_direct.py
├── judge/
│   ├── pipeline.py       # shared judge loop: build prompt -> call model -> parse scores
│   ├── backend_local.py  # Qwen2.5 via Ollama (OpenAI-compatible endpoint)
│   ├── backend_api.py    # large judge: Groq (Llama-3.3-70B) and Gemini backends
│   └── parse.py          # JSON + regex fallback parser for model output
├── eval/
│   ├── correlation.py    # Spearman / Kendall-tau / Pearson vs. human scores
│   ├── agreement.py      # judge-vs-judge agreement + mean score difference
│   ├── cost.py           # tokens, latency, throughput per summary
│   ├── bootstrap.py      # bootstrap confidence intervals for correlation
│   ├── plots.py          # cost-vs-quality and correlation-bar figures
│   └── make_tables.py    # emits report/tables.tex (booktabs) from the result CSVs
├── results/              # generated – scores, tables, plots
├── report/               # LNCS paper source (main.tex, references.bib, tables.tex)
├── config.example.yaml   # copy to config.yaml and fill in (see Configuration)
├── requirements.txt
└── README.md
```

> `data/subset.jsonl` and everything under `results/` are **generated artifacts**.
> The subset is seeded, so re-running `preprocess.py` reproduces the identical file.

---

## 4. Setup

Requires **Python 3.10–3.13**.

```bash
# 1. clone
git clone https://github.com/umitarslann9/CENG-467-Term-Project.git
cd CENG-467-Term-Project

# 2. virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. dependencies
pip install -r requirements.txt
```

### Small judge – Ollama (local)

Install [Ollama](https://ollama.com/download), then pull the small model:

```bash
ollama pull qwen2.5:3b
```

Ollama serves an OpenAI-compatible API on `http://localhost:11434/v1`, which the
small judge and the zero-shot baseline use.

### Large judge – Gemini API key

Create a key at <https://aistudio.google.com/apikey> and expose it as an environment
variable, or put it in a git-ignored `.env` file at the repo root:

```bash
# .env
GEMINI_API_KEY=...
# GROQ_API_KEY=gsk_...   # optional, only if using the free Groq backend
```

Keys are **never** stored in `config.yaml`; they are read from the environment / `.env`.
Full-scale runs need the (cheap) pay-as-you-go tier; the free tier is heavily
rate-limited. To run completely free instead, set `large_judge.provider: groq` in
`config.yaml` and use a free Groq key from <https://console.groq.com>.

### Configuration

```bash
cp config.example.yaml config.yaml
```

`config.yaml` selects the models and subset size (large judge provider/model, the
local Ollama model, BERTScore device, and the seeded subset of SummEval). Defaults:
large judge `gemini-2.5-flash` (Gemini API), small judge `qwen2.5:3b` (Ollama),
100 articles × 8 summaries, seed 42.

---

## 5. How to run

The pipeline runs **stage by stage**; each stage writes to disk so later stages can
be re-run without repeating expensive work.

```bash
# Stage 1 – data (run once)
python data/download.py               # -> data/summeval_raw.jsonl
python data/preprocess.py             # -> data/subset.jsonl

# Stage 2 – cheap baselines
python baselines/rouge_baseline.py        # -> results/rouge.jsonl
python baselines/bertscore_baseline.py    # -> results/bertscore.jsonl  (CPU)

# Stage 3 – small judge + zero-shot baseline (local, via Ollama)
python baselines/zeroshot_direct.py       # -> results/zeroshot.jsonl
python judge/pipeline.py --judge small    # -> results/judge_small.jsonl

# Stage 4 – large judge (Gemini API)
python judge/pipeline.py --judge large --limit 5   # smoke test first
python judge/pipeline.py --judge large             # -> results/judge_large.jsonl

# Stage 5 – evaluation + analysis
python eval/correlation.py            # Spearman / Kendall-tau vs. human, per dimension
python eval/agreement.py              # large-vs-small agreement
python eval/cost.py                   # cost / speed table
python eval/bootstrap.py              # confidence intervals on the correlations
python eval/plots.py                  # figures for the report
python eval/make_tables.py            # report/tables.tex from the result CSVs
```

**Tip:** every model script accepts `--limit N` to run on the first N records, and
the judges/zero-shot accept `--overwrite` to ignore existing output and start fresh.
Long runs write after every record and **resume** automatically if interrupted.

---

## 6. Reproducibility notes

- The SummEval subset is selected with a **fixed seed** (`config.yaml -> subset.seed`),
  so every run evaluates identical inputs.
- All decoding uses **temperature 0** for deterministic scores.
- All prompts are committed under `prompts/` – judge behavior is fully traceable.
- The judge model name is recorded in every `results/judge_*.jsonl` row.
- The judge's *faithfulness* dimension is mapped to SummEval's *consistency* for
  correlation; the other three dimension names align directly.

---

## 7. Known risks / open issues

- **API limits / cost** – the large judge uses the Gemini API; a client-side rate
  limiter (`rpm_limit`) keeps requests under the cap. The free tier is heavily
  rate-limited, so full runs use the (very cheap) pay-as-you-go tier; a free Groq
  backend is provided as an alternative (`provider: groq`).
- **Malformed small-model output** – handled by JSON-first parsing with a regex
  fallback and limited re-queries (see `judge/parse.py`).
- **Noisy correlation on a modest subset** – addressed with bootstrap confidence
  intervals (`eval/bootstrap.py`).
- **Long source articles** – lightly truncated to fit the small model's context.

---

## 8. Report

The scientific write-up (LNCS format) lives in `report/`. Result tables are generated
automatically from the result CSVs by `eval/make_tables.py` (into `report/tables.tex`,
which `main.tex` `\input`s), so the numbers in the paper match the computed results
exactly.
