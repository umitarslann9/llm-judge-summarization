# Large vs. Small LLMs as Evaluators of Abstractive Summarization

**CENG 467 – Natural Language Understanding and Generation – Term Project**
Ümit Arslan (290201003) · Group 24 · Proposal ID P-09

---

## 1. What this project does

This project tests whether a **small, open-source LLM** can replace a **large proprietary LLM**
as an automatic *judge* for abstractive summarization quality.

The "LLM-as-a-Judge" setup is widely used because large models (GPT-4, Claude) align with
human quality judgments far better than ROUGE or BERTScore – but they are expensive,
rate-limited, and closed-source. The core question here:

> **Does swapping a large judge for a small open judge meaningfully hurt evaluation quality, and by how much?**

This is an **evaluation task**, not a generation task. The system does **not** write summaries.
It scores summaries that already exist in the SummEval benchmark and measures how well
those scores correlate with human ratings.

**Input:** `(source document, machine-generated summary, scoring rubric)`
**Output:** four numeric scores – *faithfulness, coherence, relevance, fluency* – each preceded
by a short Chain-of-Thought (CoT) justification.

---

## 2. Components

| Component | Role |
|---|---|
| **Baseline 1 – ROUGE** (ROUGE-1/2/L) | Classical n-gram overlap vs. reference summaries |
| **Baseline 2 – BERTScore** | Embedding-based semantic similarity vs. reference summaries |
| **Baseline 3 – Zero-shot Direct Scoring** | Small model, single number, *no rubric, no CoT* – isolates prompt-design effect |
| **Small Judge** | Qwen2.5 (local) + full rubric + CoT |
| **Large Judge** | GPT-4 / Claude (API) + full rubric + CoT – reference judge |

All five run on the **same fixed SummEval subset** so comparisons are apples-to-apples.

---

## 3. Repository structure

```
CENG-467-Term-Project/
├── data/                 # SummEval download + preprocessing; fixed evaluation subset
│   ├── download.py       # pulls mteb/summeval from the HF Hub
│   ├── preprocess.py     # builds the fixed ~100-article subset (seeded), writes subset.jsonl
│   └── subset.jsonl      # generated – the frozen evaluation set (do not edit by hand)
├── prompts/              # prompt templates
│   ├── rubric_cot.txt    # 4-dimension rubric + CoT instruction (used by both judges)
│   └── direct_score.txt  # minimal "score this summary" prompt (Baseline 3)
├── baselines/
│   ├── rouge_baseline.py
│   ├── bertscore_baseline.py
│   └── zeroshot_direct.py
├── judge/
│   ├── pipeline.py       # shared judge loop: build prompt -> call model -> parse scores
│   ├── backend_local.py  # Qwen2.5 via transformers
│   └── backend_api.py    # GPT-4 / Claude via API
├── eval/
│   ├── correlation.py    # Spearman / Kendall-tau / Pearson vs. human scores
│   ├── agreement.py      # judge-vs-judge agreement + mean score difference
│   ├── cost.py           # API tokens, GPU-hours, latency per summary
│   └── bootstrap.py      # bootstrap confidence intervals for correlation
├── results/              # generated – scores, tables, plots
├── config.example.yaml   # copy to config.yaml and fill in (see Configuration)
├── requirements.txt
└── README.md
```

> `data/subset.jsonl` and everything under `results/` are **generated artifacts**.
> The subset is seeded, so anyone re-running `preprocess.py` gets the identical file.

---

## 4. Setup

Requires **Python 3.10+**. A CUDA GPU is recommended for the small judge and BERTScore,
but the ROUGE baseline and all evaluation scripts run fine on CPU.

```bash
# 1. clone
git clone https://github.com/nadezhda648/CENG-467-Term-Project.git
cd CENG-467-Term-Project

# 2. virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. dependencies
pip install -r requirements.txt
```

### Configuration

API keys and model names live in `config.yaml` (git-ignored). Create it from the template:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml`:

```yaml
large_judge:
  provider: openai          # openai | anthropic
  model: gpt-4o
  api_key_env: OPENAI_API_KEY   # key is read from this environment variable, never hard-coded
small_judge:
  model: Qwen/Qwen2.5-7B-Instruct
  device: cuda              # cuda | cpu
subset:
  n_articles: 100
  seed: 42                  # fixes the SummEval subset – keep constant for reproducibility
```

Set the API key as an environment variable (do **not** put the key itself in the file):

```bash
export OPENAI_API_KEY="sk-..."        # or ANTHROPIC_API_KEY
```

---

## 5. How to run

The pipeline is meant to be run **stage by stage** – each stage writes to disk so later
stages can be re-run without repeating expensive work.

```bash
# Stage 1 – data (run once)
python data/download.py
python data/preprocess.py            # -> data/subset.jsonl

# Stage 2 – cheap baselines (no API, no GPU needed for ROUGE)
python baselines/rouge_baseline.py        # -> results/rouge.jsonl
python baselines/bertscore_baseline.py    # -> results/bertscore.jsonl

# Stage 3 – small judge + zero-shot baseline (local GPU)
python baselines/zeroshot_direct.py       # -> results/zeroshot.jsonl
python judge/pipeline.py --judge small    # -> results/judge_small.jsonl

# Stage 4 – large judge (uses API budget – run last, after a small test batch)
python judge/pipeline.py --judge large --limit 5   # smoke test first
python judge/pipeline.py --judge large             # -> results/judge_large.jsonl

# Stage 5 – evaluation + analysis
python eval/correlation.py           # Spearman / Kendall-tau vs. human, per dimension
python eval/agreement.py             # large-vs-small agreement
python eval/cost.py                  # cost / speed table
python eval/bootstrap.py             # confidence intervals on the correlations
```

**Tip:** every model script accepts `--limit N` to run on the first N instances – use it
to validate a script end-to-end before a full (or paid) run.

---

## 6. Reproducibility notes

- The SummEval subset is selected with a **fixed seed** (`config.yaml -> subset.seed`),
  so every run evaluates identical inputs.
- Small-judge decoding defaults to **temperature 0** for deterministic scores; sampling
  runs are opt-in and logged separately.
- All prompts are committed under `prompts/` – judge behavior is fully traceable.
- `requirements.txt` pins exact versions.
- The large-judge API model name and date are recorded in `results/judge_large.jsonl`
  metadata so the run can be cited without re-paying for it.

---

## 7. Known risks / open issues

- **API budget** – the large judge is the only paid component; the subset size caps the cost.
- **Malformed small-model output** – handled by structured JSON output + a regex fallback
  parser with limited retries (see `judge/pipeline.py`).
- **Noisy correlation on a small subset** – addressed with bootstrap confidence intervals.
- **Long source articles** – may approach the small model's context limit; light truncation
  of the article body is applied when needed.

---

## 8. Status

Early stage – repository initialized, implementation starting. See Section 11 of the
progress report for the week-by-week plan. Development is tracked through incremental commits.
