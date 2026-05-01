# Gaia Intelligent Query Pipeline
### Natural Language Interface for the ESA Gaia DR3 Dataset

**Guillem Masdemont Serra · Pietro Sestito · Plabon Shaha**  
*FRI Natural Language Processing Course 2026 — Advisors: Slavko Žitnik*

---

## Goal

The [ESA Gaia Data Release 3](https://archives.esac.esa.int/gaia) catalog contains ~1.8 billion stellar sources and represents the most detailed three-dimensional map of the Milky Way ever assembled. Accessing it requires ADQL, a SQL-based query language that is a significant barrier for non-specialists, students, and casual science enthusiasts. This project builds a **Retrieval-Augmented Generation (RAG) pipeline** that lets anyone query the Gaia database in plain English. You type a question like *"Show me red dwarfs near Barnard's star"* and the system handles the internal procedure to extract SQL from text and get a .csv file. 

---

## Pipeline (so far)

```
User natural-language query
        │
        ▼
┌─────────────────────┐
│   LLM Query Parser  │  Qwen2.5-7B-Instruct (vLLM)
│  (JSON intent)      │  Intent + coordinates + filters + columns
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Teacher Validator  │  Checks intent, coordinate ranges, columns,
│  (rule-based)       │  and provides the tool to call
└──────────┬──────────┘   (up to 3 retries with error feedback)
           │
           ▼
┌─────────────────────┐
│   ADQL Builder      │  Deterministic intent → ADQL translation
│                     │  (cone search, JOIN, ORDER BY, null guards…)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Cost Judge        │  Check if the query is feasible or too big
│  cheap/mod/exp/danger│  Blocks dangerous queries, auto-optimises
└──────────┬──────────┘  (random sampling, cone shrink, TOP cap)
           │
           ▼
┌─────────────────────┐
│  Gaia TAP Executor  │  We send a async job to astroquery
│  (retry + backoff)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   HTML Report       │  Present results in Sky map · Celestial sphere 
│   (display_html.py) │  Histograms · Colour-magnitude diagram · Stats cards
└─────────────────────┘
```

## Example: "Provide me the brightest stars"

The pipeline parses the query, builds a validated ADQL query, confirms the cost is cheap, fetches 634 sources from Gaia DR3, and produces a full visual report.

![alt text](image.png)
![alt text](image-1.png)

## Future directions

- **Evaluation harness** — Automated scoring of generated ADQL against the ground-truth queries in `src/dataset/queries_1000.csv` (simple / medium / complex tiers).
- **Retrieval-augmented generation** — Embed Gaia DR3 documentation and create more difficult queries. This is planned to be done by integrating an LLM capable of joining several queries into one. 

## Repository layout

```
src/
  astronomy_query.ipynb   # Main pipeline notebook
  display_html.py         # HTML report generator
  gaia_report.html        # Example report (brightest stars query)
  dataset/
    queries_100_v2.csv    # Benchmark: 100 NL→ADQL pairs (simple/medium/complex)
    queries_1000.csv      # Extended benchmark set
report/
  report.tex              # Project interim report (LaTeX)
  report.pdf
docs/
  sky_scatter.png         # Extracted visualisation examples
  colour_histogram.png
  statistical_plots.png
```
