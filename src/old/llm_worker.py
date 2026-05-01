"""
llm_worker.py
─────────────────────────────────────────────────────────────────────────────
Runs on the GPU compute node via sbatch.
Reads a query from outputs/current_query.txt,
runs vLLM + Gaia, saves results to outputs/query_result.json.
─────────────────────────────────────────────────────────────────────────────
"""

# ── SSL fix (must be first) ───────────────────────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

import json
import sys
import numpy as np

BASE_DIR = "/d/hpc/projects/onj_fri/no-language-processors-v2"
QUERY_FILE  = f"{BASE_DIR}/outputs/current_query.txt"
OUTPUT_FILE = f"{BASE_DIR}/outputs/query_result.json"
STATUS_FILE = f"{BASE_DIR}/outputs/job_status.txt"

os.makedirs(f"{BASE_DIR}/outputs", exist_ok=True)


def write_status(msg: str):
    """Write current status so Streamlit can show progress."""
    with open(STATUS_FILE, "w") as f:
        f.write(msg)
    print(f"[status] {msg}")


# ── Load query ────────────────────────────────────────────────────────────────
write_status("reading_query")

if not os.path.exists(QUERY_FILE):
    write_status("error: no query file found")
    sys.exit(1)

with open(QUERY_FILE) as f:
    user_query = f.read().strip()

print(f"[worker] Query: {user_query}")

# ── Load vLLM model ───────────────────────────────────────────────────────────
write_status("loading_model")

from vllm import LLM, SamplingParams

MODEL_ID             = os.environ.get("MODEL_ID",           "Qwen/Qwen2.5-7B-Instruct")
HF_CACHE             = os.environ.get("HF_CACHE",           f"{BASE_DIR}/hf_cache")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
DTYPE                = os.environ.get("DTYPE",              "float16")

os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
os.environ.setdefault("HF_HOME",            HF_CACHE)

print(f"[worker] Loading model '{MODEL_ID}'...")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    dtype=DTYPE,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enforce_eager=True,   # skip 26s CUDA graph capture
)
print("[worker] Model loaded.")

# ── System prompt + parse ─────────────────────────────────────────────────────
write_status("parsing_query")

SYSTEM_PROMPT = """You are an astronomy query parser.
Convert user queries into structured JSON.
Supported intents: cone_search, cone_search_with_join, color_histogram, velocity_computation
Return ONLY valid JSON — no markdown fences, no extra text.
Schema: {"intent": "...", "ra": float, "dec": float, "radius": float, "columns": [list], "limit": int}
Valid Gaia DR3 columns: ra, dec, source_id, parallax, parallax_error, pmra, pmdec, radial_velocity,
  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, teff_gspphot, logg_gspphot, mh_gspphot
If unknown values use RA=266.4, DEC=-29.0. Default radius: 1.0. Default limit: 1000."""

SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=256)

prompt = (
    "<|im_start|>system\n"
    f"{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n"
    f"{user_query}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

outputs = llm.generate([prompt], SAMPLING_PARAMS)
raw = outputs[0].outputs[0].text.strip()

if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
    raw = raw.strip()

try:
    q = json.loads(raw)
    print(f"[worker] Parsed: {json.dumps(q, indent=2)}")
except json.JSONDecodeError as e:
    write_status(f"error: LLM returned invalid JSON: {raw}")
    sys.exit(1)

# ── Build ADQL ────────────────────────────────────────────────────────────────
write_status("building_adql")

_COLUMN_ALIASES = {
    "magnitude": "phot_g_mean_mag", "g_magnitude": "phot_g_mean_mag",
    "g_mag": "phot_g_mean_mag", "bp_magnitude": "phot_bp_mean_mag",
    "rp_magnitude": "phot_rp_mean_mag", "color": "bp_rp", "colour": "bp_rp",
    "temperature": "teff_gspphot", "teff": "teff_gspphot",
    "proper_motion_ra": "pmra", "proper_motion_dec": "pmdec",
}

intent  = q["intent"]
ra      = q["ra"]
dec     = q["dec"]
radius  = q["radius"]
limit   = q.get("limit", 1000)
columns = [_COLUMN_ALIASES.get(c, c) for c in (q.get("columns") or ["ra", "dec", "parallax"])]
cone    = f"1=CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS', {ra}, {dec}, {radius}))"

if intent == "cone_search":
    cols = ", ".join(columns)
    adql = f"SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {cone}"

elif intent == "cone_search_with_join":
    cone_j = f"1=CONTAINS(POINT('ICRS', g.ra, g.dec),CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    adql = (f"SELECT TOP {limit} g.source_id, g.ra, g.dec, d.r_med_geo "
            f"FROM gaiadr3.gaia_source AS g "
            f"JOIN gaiadr3.geometric_distance_prior AS d ON g.source_id = d.source_id "
            f"WHERE {cone_j}")

elif intent == "color_histogram":
    adql = (f"SELECT TOP {limit} bp_rp FROM gaiadr3.gaia_source "
            f"WHERE bp_rp IS NOT NULL AND {cone}")

elif intent == "velocity_computation":
    adql = (f"SELECT TOP {limit} ra, dec, parallax, pmra, pmdec, radial_velocity "
            f"FROM gaiadr3.gaia_source "
            f"WHERE parallax > 0 AND radial_velocity IS NOT NULL AND {cone} "
            f"ORDER BY parallax DESC")
else:
    cols = ", ".join(columns)
    adql = f"SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {cone}"

print(f"[worker] ADQL:\n{adql}")

# ── Run Gaia query ────────────────────────────────────────────────────────────
write_status("querying_gaia")

import time
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

last_exc = None
for attempt in range(3):
    try:
        job = Gaia.launch_job_async(adql, verbose=False)
        df  = job.get_results().to_pandas()
        break
    except Exception as exc:
        last_exc = exc
        if attempt < 2:
            wait = 15 * (2 ** attempt)
            print(f"[worker] attempt {attempt+1} failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)
else:
    write_status(f"error: Gaia query failed: {last_exc}")
    sys.exit(1)

print(f"[worker] {len(df)} rows returned.")

# ── Compute extra columns for velocity intent ─────────────────────────────────
if intent == "velocity_computation":
    valid = df["parallax"].notna() & (df["parallax"] > 0)
    df_v  = df[valid].copy()
    if len(df_v) > 0:
        d  = 1000.0 / df_v["parallax"]
        mu = np.sqrt(df_v["pmra"]**2 + df_v["pmdec"]**2)
        df_v["velocity_km_s"] = np.sqrt(df_v["radial_velocity"].fillna(0)**2 + (4.74*mu*d)**2)
        df = df_v

# ── Save results ──────────────────────────────────────────────────────────────
write_status("saving_results")

# Replace NaN with None so JSON serialises cleanly
output = {
    "adql":   adql,
    "intent": intent,
    "ra":     ra,
    "dec":    dec,
    "radius": radius,
    "query":  user_query,
    "rows":   df.where(df.notna(), other=None).to_dict(orient="records"),
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f)

write_status("done")
print(f"[worker] Saved {len(df)} rows → {OUTPUT_FILE}")
