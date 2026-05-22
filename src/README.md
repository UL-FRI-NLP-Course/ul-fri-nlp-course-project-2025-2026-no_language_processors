# Source Code Guide

This directory contains everything needed to run and evaluate the Gaia Intelligent Query Pipeline.

**There are two entry points, both in `pipeline/`:**

- **`main.ipynb`** — run a query end-to-end through the pipeline and inspect results
- **`evaluate_pipeline.ipynb`** — batch-evaluate the pipeline against the benchmark dataset

All other modules are imported by these notebooks; you do not need to call them directly.

> **Reminder:** The pipeline requires GPU + the Qwen2.5-7B model. It cannot be run on a laptop. See the root README for HPC setup instructions.

---

## Directory Overview

```
src/
├── pipeline/           ← Core pipeline code (start here)
│   ├── main.ipynb            Entry point: run a query end-to-end
│   ├── evaluate_pipeline.ipynb  Batch evaluation against the benchmark
│   ├── pipeline.py           Top-level router (calls simple or complex path)
│   ├── config.py             Shared constants, env setup, SSL workarounds
│   ├── model.py              vLLM model loader (singleton)
│   ├── router.py             LLM-based query complexity classifier
│   ├── parser.py             LLM intent parser + deterministic ADQL builder
│   ├── simple_pipeline.py    Orchestrates the single-query path
│   ├── complex_pipeline.py   Orchestrates the multi-step path
│   ├── cost_judge.py         Query cost evaluator and auto-optimiser
│   ├── eval_metrics.py       BLEU, token-F1, structural similarity scoring
│   └── display_html.py       Interactive HTML report generator
├── dataset/            ← Benchmark data and evaluation results
│   ├── gaia_eval_dataset.csv     40 NL→ADQL pairs (4 intent types)
│   ├── eval_resultss.csv         Per-query evaluation results
│   └── generated_dataset.ipynb   Notebook used to generate the dataset
├── docker/             ← HPC deployment
│   ├── vllm.def              Apptainer container recipe (CUDA 12.4 + vLLM)
│   └── vllm.sh               SLURM batch job script
├── output/             ← Pre-generated outputs (read-only examples)
│   ├── displayed_html/       Example HTML reports for 4 sample queries
│   └── pipeline_outputs/     JSON execution logs for ~16 queries
├── display_html.py     ← Standalone copy of the HTML generator
└── gaia_report.html    ← Example output: brightest stars query
```

---

## Module Reference

### `pipeline.py` — Entry point

```python
from pipeline import routed_pipeline

result = routed_pipeline("Find 700 nearby red stars")
# Returns:
# {
#   'output_json': {...},   # metadata, generated ADQL, cost verdict
#   'results': {1: DataFrame, ...}  # query results
# }
```

Calls `router.py` to classify the query, then dispatches to `simple_pipeline.py` or `complex_pipeline.py`.

---

### `config.py` — Shared configuration

Loaded implicitly by all other modules. Sets:
- HuggingFace cache directory (`../hf_cache`)
- Pipeline output directory (`./pipeline_outputs`)
- `MAX_RETRIES = 3` for all retry loops
- SSL certificate bypass (needed on ARNES HPC due to institutional firewall)

---

### `model.py` — LLM singleton

Loads **Qwen2.5-7B-Instruct** via vLLM once per session and caches it. Exports three sampling configs:

| Export | Temperature | Max tokens | Used by |
|--------|-------------|------------|---------|
| `SAMPLING_PARAMS` | 0.0 | 512 | Router, parser, cost judge |
| `JUDGE_SAMPLING_PARAMS` | 0.0 | 512 | Cost judge |
| `DECOMPOSER_SAMPLING_PARAMS` | 0.0 | 1024 | Complex pipeline decomposer |

Set `TENSOR_PARALLEL_SIZE` env var to use multiple GPUs.

---

### `router.py` — Complexity classifier

```python
from router import route_query

route_query("Find stars in Orion and also in Andromeda")
# → {'complexity': 'complex', 'reason': 'Two distinct sky regions'}
```

Prompts the LLM to decide between **simple** (single analysis, single region) and **complex** (multiple outputs, multiple regions, or sequential dependencies). Falls back to `simple` on parse errors.

---

### `parser.py` — Intent parser and ADQL builder

The heart of the pipeline. Two steps:

**Step 1 — Parse:** LLM converts the query into a structured JSON intent:

```json
{
  "intent": "stellar_population",
  "ra": 83.82,
  "dec": -5.39,
  "radius": 1.0,
  "columns": ["source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp"],
  "filters": {"bp_rp": {">": 1.5}},
  "join_table": null,
  "limit": 500
}
```

Supported intents: `cone_search`, `hr_diagram`, `stellar_population`, `variability_search`, `velocity_kinematics`, `cross_match`, `nearest_neighbor`, `sky_wide_superlative`.

**Step 2 — Build:** `build_adql(parsed_json)` converts the validated JSON to ADQL deterministically. Handles cone geometry (`CONTAINS(POINT(...), CIRCLE(...))`), table JOINs, null guards, and absolute magnitude computations.

---

### `simple_pipeline.py` — Simple path orchestrator

Runs the full simple pipeline with a retry loop (up to `MAX_RETRIES = 3`):

1. Parse query → JSON intent
2. Validate JSON (field types, coordinate ranges, valid column names)
3. Build ADQL
4. Validate ADQL syntax
5. Evaluate cost (fast precheck + LLM judge)
6. Auto-optimise if needed
7. Execute via Gaia TAP
8. Save JSON log to `pipeline_outputs/`

---

### `complex_pipeline.py` — Complex path orchestrator

Decomposes multi-part queries into an execution plan and runs it.

**Decomposition:** The LLM produces a plan with one of three composition types:

| Type | Description |
|------|-------------|
| `merged` | Two or more analyses on the same sky region → combined into one ADQL |
| `parallel` | N independent sub-queries on different regions → run separately |
| `sequential` | Step N feeds its `source_id` results into step N+1 |

The plan is validated for structure, valid intents, and dependency ordering before execution. On step failure, the outer loop re-decomposes (up to `PLAN_RETRIES = 2`).

---

### `cost_judge.py` — Cost evaluator

Prevents runaway queries from overloading the Gaia TAP server.

**Fast precheck** (rule-based, no LLM):
- Missing WHERE clause → `dangerous`
- Missing TOP → `expensive`
- TOP > 100,000 → `expensive`

**LLM judge** (0–100 score):
- Full-sky scans without spatial filters → high cost
- Cone radius > 2° → moderate cost
- ORDER BY on non-indexed columns → moderate cost
- Weak parallax filter → adds to cost

**Auto-optimiser** (deterministic rewrites):
- Caps TOP ≤ 5,000 on sky-wide queries
- Shrinks cone radius > 3° → 2°
- Strengthens `parallax > 0` → `parallax > 1`
- Injects `MOD(random_index, N) = 0` for unfiltered large queries

Verdicts: `cheap` → `moderate` → `expensive` → `dangerous` (blocks execution).

---

### `eval_metrics.py` — Evaluation metrics

Used by `evaluate_pipeline.ipynb` to score generated ADQL against ground truth:

| Function | Description |
|----------|-------------|
| `bleu_score(hyp, ref)` | 4-gram BLEU overlap (0–1) |
| `exact_match(hyp, ref)` | Binary match after normalisation |
| `token_f1(hyp, ref)` | Precision/recall of SQL tokens |
| `structural_match(hyp, ref)` | Structural similarity (clauses, joins, filters) |

---

### `display_html.py` — HTML report generator

Takes a pandas DataFrame of Gaia results and generates a self-contained HTML report with:

- **Sky scatter plot** — RA/Dec distribution with G-magnitude colour coding (RA axis inverted per astronomical convention)
- **Colour histogram** — BP-RP colour index distribution
- **HR diagram** — absolute magnitude vs. BP-RP colour (computed from parallax)
- **Supplementary plots** — proper motion, magnitude distributions
- **Statistics cards** — median, mean, σ for key columns

All figures are embedded as base64 PNGs; Plotly charts are included as interactive JSON. The output is a single `.html` file with no external dependencies.

---

## Dataset

`dataset/gaia_eval_dataset.csv` contains 40 NL→ADQL pairs covering four intent types:

| Intent | Count |
|--------|-------|
| `cone_search` | 10 |
| `hr_diagram` | 10 |
| `stellar_population` | 10 |
| `variability_search` | 10 |

Each row has: `query` (natural language), `adql` (ground-truth ADQL), `intent`, and metadata fields.

`dataset/eval_resultss.csv` contains the per-query results from running the pipeline against this benchmark, with columns for each metric.

---

