"""
astronomy_query.py
─────────────────────────────────────────────────────────────────────────────
Astronomy query parser using vLLM loaded in-process (no HTTP server).
The model is loaded once at startup and reused for all queries.

Usage:
  python astronomy_query.py
  python astronomy_query.py "Stars near Orion within 0.5 degrees"

Environment variables (all optional — defaults shown):
  MODEL_ID       HuggingFace model repo  (default: Qwen/Qwen2.5-7B-Instruct)
  HF_CACHE       HuggingFace cache dir   (default: ~/.cache/huggingface)
  TENSOR_PARALLEL_SIZE   number of GPUs  (default: 1)
─────────────────────────────────────────────────────────────────────────────
"""



# ── SSL fix for ARNES cluster (self-signed certificate in proxy) ──────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
os.environ.setdefault("CURL_CA_BUNDLE", "")      # fixes some astroquery internals
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")  # fixes requests-based calls

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed on HPC headless nodes
import matplotlib.pyplot as plt

from vllm import LLM, SamplingParams

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID             = os.environ.get("MODEL_ID",           "Qwen/Qwen2.5-7B-Instruct")
HF_CACHE             = os.environ.get("HF_CACHE",           os.path.expanduser("~/.cache/huggingface"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
DTYPE                = os.environ.get("DTYPE",              "float16")

os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
os.environ.setdefault("HF_HOME",            HF_CACHE)

# ── Load model once at startup ────────────────────────────────────────────────
print(f"[init] Loading model '{MODEL_ID}' (tensor_parallel_size={TENSOR_PARALLEL_SIZE}) …")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    dtype=DTYPE,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,     # required for some HF models
)
print("[init] Model loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an astronomy query parser.

Convert user queries into structured JSON.

Supported intents:
- cone_search
- cone_search_with_join
- color_histogram
- velocity_computation

Return ONLY valid JSON — no markdown fences, no extra text.

Schema:

{
  "intent": "...",
  "ra": float,
  "dec": float,
  "radius": float,
  "columns": [list],
  "limit": int
}

Valid Gaia DR3 column names (use ONLY these):
  ra, dec, source_id, parallax, parallax_error,
  pmra, pmdec, radial_velocity,
  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
  teff_gspphot, logg_gspphot, mh_gspphot

If unknown values (like galactic center), use:
RA=266.4, DEC=-29.0

Default radius: 1.0 degree.
Default limit: 1000."""

SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,    # deterministic — we need well-formed JSON
    max_tokens=256,
)


# ── Query parsing ─────────────────────────────────────────────────────────────

def _build_prompt(user_query: str) -> str:
    """Format a chat prompt using Qwen2.5-Instruct's ChatML template."""
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_query(user_query: str) -> dict:
    """Run the natural-language query through the local LLM and parse the returned JSON."""
    prompt = _build_prompt(user_query)
    outputs = llm.generate([prompt], SAMPLING_PARAMS)
    raw = outputs[0].outputs[0].text.strip()

    # Strip optional markdown code fences that some models add despite the prompt
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output:\n{raw}") from exc


# ── ADQL builder ──────────────────────────────────────────────────────────────

_COLUMN_ALIASES = {
    "magnitude":       "phot_g_mean_mag",
    "g_magnitude":     "phot_g_mean_mag",
    "g_mag":           "phot_g_mean_mag",
    "bp_magnitude":    "phot_bp_mean_mag",
    "rp_magnitude":    "phot_rp_mean_mag",
    "color":           "bp_rp",
    "colour":          "bp_rp",
    "temperature":     "teff_gspphot",
    "teff":            "teff_gspphot",
    "proper_motion_ra":  "pmra",
    "proper_motion_dec": "pmdec",
}

def build_adql(q: dict) -> str:
    intent  = q["intent"]
    ra      = q["ra"]
    dec     = q["dec"]
    radius  = q["radius"]
    limit   = q.get("limit", 1000)
    columns = [_COLUMN_ALIASES.get(c, c) for c in (q.get("columns") or ["ra", "dec", "parallax"])]

    # Shared cone predicate — single line, no extra whitespace
    cone = (
        f"1=CONTAINS(POINT('ICRS', ra, dec),"
        f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )

    if intent == "cone_search":
        cols = ", ".join(columns)
        return (
            f"SELECT TOP {limit} {cols} "
            f"FROM gaiadr3.gaia_source "
            f"WHERE {cone}"
        )

    elif intent == "cone_search_with_join":
        cone_join = (
            f"1=CONTAINS(POINT('ICRS', g.ra, g.dec),"
            f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
        )
        return (
            f"SELECT TOP {limit} g.source_id, g.ra, g.dec, d.r_med_geo "
            f"FROM gaiadr3.gaia_source AS g "
            f"JOIN gaiadr3.geometric_distance_prior AS d "   # correct DR3 table
            f"ON g.source_id = d.source_id "
            f"WHERE {cone_join}"
        )

    elif intent == "color_histogram":
        return (
            f"SELECT TOP {limit} bp_rp "
            f"FROM gaiadr3.gaia_source "
            f"WHERE bp_rp IS NOT NULL AND {cone}"           # add spatial filter
        )

    elif intent == "velocity_computation":
        return (
            f"SELECT TOP {limit} ra, dec, parallax, pmra, pmdec, radial_velocity "
            f"FROM gaiadr3.gaia_source "
            f"WHERE parallax > 0 "
            f"AND radial_velocity IS NOT NULL "             # avoid nulls early
            f"AND {cone} "                                  # add spatial filter
            f"ORDER BY parallax DESC"
        )

    else:
        raise ValueError(f"Unknown intent: '{intent}'")

# ── Gaia query execution ───────────────────────────────────────────────────────

def run_query(adql: str, retries: int = 3, backoff: float = 15.0):
    """Submit an ADQL query to the Gaia TAP service and return a pandas DataFrame."""
    import time
    from astroquery.gaia import Gaia

    # Correct way to disable SSL in astroquery's TAP handler
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    last_exc = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job_async(adql, verbose=False)
            return job.get_results().to_pandas()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"[run_query] attempt {attempt + 1} failed ({exc}); retrying in {wait:.0f}s …")
                time.sleep(wait)
    raise last_exc


# ── Post-processing ───────────────────────────────────────────────────────────

def plot_histogram(df, output_path: str = "color_histogram.png") -> None:
    """Plot and save a BP-RP colour histogram from a Gaia results DataFrame."""
    bp_rp = df["bp_rp"].dropna()

    if bp_rp.empty:
        print("[plot_histogram] No BP-RP data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(bp_rp, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel("BP – RP  (mag)", fontsize=12)
    ax.set_ylabel("Count",          fontsize=12)
    ax.set_title("Gaia star colour distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_histogram] Saved → {output_path}")


def compute_velocity(df) -> None:
    """
    Compute total space velocity from proper motions + radial velocity.

    v_tangential = 4.74 * mu_total * d     [km/s]
    d = 1000 / parallax                    [pc]   (parallax in mas)
    v_total      = sqrt(v_r^2 + v_t^2)
    """
    valid = df["parallax"].notna() & (df["parallax"] > 0)
    if valid.sum() == 0:
        print("[compute_velocity] No rows with valid parallax.")
        return

    df = df[valid].copy()

    d  = 1000.0 / df["parallax"]
    mu = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)
    vt = 4.74 * mu * d
    vr = df["radial_velocity"].fillna(0.0)

    df["velocity_km_s"] = np.sqrt(vr ** 2 + vt ** 2)

    print("\n── Velocity statistics (km/s) ──────────────────────")
    print(df["velocity_km_s"].describe().to_string())
    print("────────────────────────────────────────────────────\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_system(user_query: str) -> None:
    print(f"\n{'='*60}")
    print(f"User:   {user_query}")

    q = parse_query(user_query)
    print(f"Parsed: {json.dumps(q, indent=2)}")

    adql = build_adql(q)
    print(f"\nADQL:\n{adql}\n")

    try:
        df = run_query(adql)
    except Exception as exc:
        print(f"[run_system] Query failed after retries: {exc}")
        return
    print(f"Rows returned: {len(df)}")

    intent = q["intent"]

    if intent == "color_histogram":
        plot_histogram(df)
    elif intent == "velocity_computation":
        compute_velocity(df)
    else:
        print(df.head().to_string())

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    
    if len(sys.argv) > 1:
        run_system(" ".join(sys.argv[1:]))
    else:
        run_system("Do a cone search at (30,30) and with a very very small radius, like 0.05")
        #run_system("What color are the stars?")
        #run_system("How fast do the closest stars move?")
    