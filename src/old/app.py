"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit frontend for the astronomy query system.
Runs on the login node — no GPU needed.
Submits GPU jobs via sbatch and displays results.
─────────────────────────────────────────────────────────────────────────────
"""

# ── SSL fix ───────────────────────────────────────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

import json
import io
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = "/d/hpc/projects/onj_fri/no-language-processors-v2"
OUTPUT_FILE = f"{BASE_DIR}/outputs/query_result.json"
QUERY_FILE  = f"{BASE_DIR}/outputs/current_query.txt"
STATUS_FILE = f"{BASE_DIR}/outputs/job_status.txt"
SBATCH_PATH = "/d/hpc/projects/onj_fri/no-language-processors-v2/sbatch_wrapper.sh"
SUBMIT_SH   = f"{BASE_DIR}/submit_llm.sh"

os.makedirs(f"{BASE_DIR}/outputs", exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Celestial Query Explorer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Cormorant+Garamond:wght@300;600&display=swap');
html, body, [data-testid="stApp"] {
    background-color: #03040a !important; color: #c8d8f0 !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stSidebar"] { background-color: #080c18 !important; border-right: 1px solid #1a2540 !important; }
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; color: #e8eeff !important; }
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background-color: #0d1528 !important; border: 1px solid #1a2540 !important;
    color: #c8d8f0 !important; font-family: 'Space Mono', monospace !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0d2040, #162d50) !important;
    border: 1px solid #4fc3f7 !important; color: #4fc3f7 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #0d2a0d, #163016) !important;
    border: 1px solid #69f0ae !important; color: #69f0ae !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
}
.adql-box {
    background: #0d1528; border: 1px solid #1a2540; border-left: 3px solid #ffd54f;
    padding: 1rem; font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #ffd54f; white-space: pre-wrap; word-break: break-all; border-radius: 2px;
}
.status-box {
    background: #0d1528; border: 1px solid #1a2540; border-left: 3px solid #4fc3f7;
    padding: 0.8rem 1rem; font-family: 'Space Mono', monospace;
    font-size: 0.72rem; color: #4fc3f7; border-radius: 2px; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

STATUS_MESSAGES = {
    "reading_query":  "📂 Reading query...",
    "loading_model":  "🤖 Loading LLM model (this takes ~40s)...",
    "parsing_query":  "🔍 Parsing natural language query...",
    "building_adql":  "🔨 Building ADQL query...",
    "querying_gaia":  "🌌 Querying Gaia DR3...",
    "saving_results": "💾 Saving results...",
    "done":           "✅ Done!",
}

def read_status() -> str:
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return f.read().strip()
    return ""

def get_job_state(job_id: str) -> str:
    """Check SLURM job state."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=5
        )
        state = result.stdout.strip()
        return state if state else "COMPLETED"
    except Exception:
        return "UNKNOWN"

def make_sky_plot(df, ra_c, dec_c, radius):
    fig = plt.figure(figsize=(9, 7), facecolor="#03040a")
    ax  = fig.add_subplot(111, facecolor="#03040a")

    # Background stars
    rng    = np.random.default_rng(42)
    bg_ra  = rng.uniform(ra_c  - radius*3, ra_c  + radius*3, 600)
    bg_dec = rng.uniform(dec_c - radius*3, dec_c + radius*3, 600)
    ax.scatter(bg_ra, bg_dec, s=rng.uniform(0.2, 1.5, 600),
               c="white", alpha=rng.uniform(0.05, 0.2, 600), zorder=1)

    # Search circle
    ax.add_patch(Circle((ra_c, dec_c), radius,
                        fill=False, edgecolor="#4fc3f7",
                        linewidth=0.8, linestyle="--", alpha=0.5, zorder=3))
    ax.plot(ra_c, dec_c, "+", color="#4fc3f7", markersize=10,
            markeredgewidth=0.8, alpha=0.7, zorder=4)

    # Query stars
    if "ra" in df.columns and "dec" in df.columns and len(df) > 0:
        ra_col  = df["ra"].values.astype(float)
        dec_col = df["dec"].values.astype(float)

        if "phot_g_mean_mag" in df.columns:
            mag      = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").fillna(15)
            mag_norm = (mag - mag.min()) / (mag.ptp() + 1e-9)
            sizes    = (1 - mag_norm) * 40 + 2
            colors   = plt.cm.Blues_r(0.3 + mag_norm * 0.5)
        else:
            sizes  = np.full(len(df), 6)
            colors = "#4fc3f7"

        ax.scatter(ra_col, dec_col, s=sizes, c=colors, alpha=0.85, zorder=5)
        ax.scatter(ra_col, dec_col, s=sizes*4, c=colors, alpha=0.07, zorder=4)

    pad = radius * 2.2
    ax.set_xlim(ra_c + pad, ra_c - pad)
    ax.set_ylim(dec_c - pad, dec_c + pad)
    ax.set_xlabel("Right Ascension (°)", color="#556080", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Declination (°)",     color="#556080", fontsize=8, fontfamily="monospace")
    ax.set_title(f"RA {ra_c:.3f}°  ·  Dec {dec_c:.3f}°  ·  r = {radius}°",
                 color="#c8d8f0", fontsize=9, fontfamily="monospace")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2540")
    ax.tick_params(colors="#556080", labelsize=7)
    ax.grid(True, color="#0d1528", linewidth=0.5, alpha=0.8)
    ax.text(0.02, 0.97, f"{len(df)} sources", transform=ax.transAxes,
            color="#4fc3f7", fontsize=7, fontfamily="monospace", va="top")
    fig.tight_layout()
    return fig

def make_histogram_plot(df):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#03040a")
    ax.set_facecolor("#03040a")
    bp_rp  = pd.to_numeric(df["bp_rp"], errors="coerce").dropna()
    counts, bins, patches = ax.hist(bp_rp, bins=80, edgecolor="none")
    norm = plt.Normalize(bins.min(), bins.max())
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.RdYlBu_r(norm(left)))
        patch.set_alpha(0.85)
    ax.set_xlabel("BP – RP (mag)", color="#556080", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Count",         color="#556080", fontsize=8, fontfamily="monospace")
    ax.set_title("Stellar Colour Distribution", color="#c8d8f0", fontsize=10, fontfamily="monospace")
    for spine in ax.spines.values(): spine.set_edgecolor("#1a2540")
    ax.tick_params(colors="#556080", labelsize=7)
    ax.grid(True, color="#0d1528", linewidth=0.5, axis="y", alpha=0.6)
    fig.tight_layout()
    return fig

def make_velocity_plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#03040a")
    for ax in axes:
        ax.set_facecolor("#03040a")
        for spine in ax.spines.values(): spine.set_edgecolor("#1a2540")
        ax.tick_params(colors="#556080", labelsize=7)
        ax.grid(True, color="#0d1528", linewidth=0.5, alpha=0.6)

    vel = pd.to_numeric(df.get("velocity_km_s", pd.Series([])), errors="coerce").dropna()
    if len(vel) > 0:
        axes[0].hist(vel.clip(0, 500), bins=60, color="#4fc3f7", alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("Total velocity (km/s)", color="#556080", fontsize=8, fontfamily="monospace")
    axes[0].set_ylabel("Count",                 color="#556080", fontsize=8, fontfamily="monospace")
    axes[0].set_title("Velocity Distribution",  color="#c8d8f0", fontsize=9, fontfamily="monospace")

    if "pmra" in df.columns and "pmdec" in df.columns:
        sample = df.sample(min(500, len(df)), random_state=42)
        pmra  = pd.to_numeric(sample["pmra"],  errors="coerce")
        pmdec = pd.to_numeric(sample["pmdec"], errors="coerce")
        speed = np.sqrt(pmra**2 + pmdec**2)
        sc    = axes[1].scatter(pmra, pmdec, c=speed, cmap="plasma", s=4, alpha=0.7)
        axes[1].axhline(0, color="#1a2540", linewidth=0.8)
        axes[1].axvline(0, color="#1a2540", linewidth=0.8)
        cbar = fig.colorbar(sc, ax=axes[1], fraction=0.03, pad=0.02)
        cbar.set_label("PM magnitude", color="#556080", fontsize=7, fontfamily="monospace")
        cbar.ax.yaxis.set_tick_params(color="#556080", labelsize=6, labelcolor="#556080")
        cbar.outline.set_edgecolor("#1a2540")
    axes[1].set_xlabel("μ_RA (mas/yr)",      color="#556080", fontsize=8, fontfamily="monospace")
    axes[1].set_ylabel("μ_Dec (mas/yr)",     color="#556080", fontsize=8, fontfamily="monospace")
    axes[1].set_title("Proper Motion Vectors", color="#c8d8f0", fontsize=9, fontfamily="monospace")
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;'>
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.5rem;color:#e8eeff;'>✦ Query Config</div>
        <div style='font-family:"Space Mono",monospace;font-size:0.6rem;color:#556080;letter-spacing:0.2em;'>GAIA DR3 · TAP SERVICE</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Natural Language Query**")
    nl_query = st.text_area("", placeholder='e.g. "Stars near Orion within 0.5 degrees"',
                             height=80, label_visibility="collapsed")
    run_btn  = st.button("⬡  Submit GPU Job", use_container_width=True)

    st.markdown("---")
    st.markdown("**Job Status**")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    status = read_status()
    if status:
        friendly = STATUS_MESSAGES.get(status, f"⚙️ {status}")
        st.markdown(f'<div class="status-box">{friendly}</div>', unsafe_allow_html=True)

    if "job_id" in st.session_state:
        job_state = get_job_state(st.session_state["job_id"])
        st.markdown(f'<div class="status-box">SLURM job {st.session_state["job_id"]}: {job_state}</div>',
                    unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 1rem 0; border-bottom:1px solid #1a2540; margin-bottom:1.5rem;'>
    <div style='font-family:"Cormorant Garamond",serif;font-size:2.8rem;font-weight:300;color:#e8eeff;'>
        Celestial Query Explorer
    </div>
    <div style='font-family:"Space Mono",monospace;font-size:0.65rem;color:#556080;letter-spacing:0.25em;margin-top:0.4rem;'>
        GAIA DR3 · LLM QUERY PARSER · ARNES HPC
    </div>
</div>""", unsafe_allow_html=True)

# ── Submit job ────────────────────────────────────────────────────────────────
if run_btn and nl_query.strip():
    # Clear old results
    for f in [OUTPUT_FILE, STATUS_FILE]:
        if os.path.exists(f): os.remove(f)

    # Write query for worker to read
    with open(QUERY_FILE, "w") as f:
        f.write(nl_query.strip())

    # Submit sbatch job
    try:
        result = subprocess.run(
            [SBATCH_PATH, SUBMIT_SH],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            st.session_state["job_id"] = job_id
            st.session_state["query"]  = nl_query.strip()
            st.success(f"✅ Job submitted: {job_id} — click Refresh to check progress")
        else:
            st.error(f"sbatch failed: {result.stderr}")
    except FileNotFoundError:
        st.error(f"sbatch not found at {SBATCH_PATH} — update SBATCH_PATH in app.py")
    except Exception as e:
        st.error(f"Failed to submit job: {e}")

elif run_btn:
    st.warning("Please enter a query first.")

# ── Show query info ───────────────────────────────────────────────────────────
if "query" in st.session_state:
    st.markdown(f"**Query:** `{st.session_state['query']}`")

# ── Show results when ready ───────────────────────────────────────────────────
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        data = json.load(f)

    df     = pd.DataFrame(data["rows"])
    adql   = data["adql"]
    intent = data["intent"]
    ra_c   = data["ra"]
    dec_c  = data["dec"]
    radius = data["radius"]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Sources",   f"{len(df):,}"),
        (c2, "Intent",    intent.replace("_", " ")),
        (c3, "RA / Dec",  f"{ra_c:.2f} / {dec_c:.2f}"),
        (c4, "Radius",    f"{radius}°"),
    ]:
        col.metric(label, value)

    # ADQL
    st.markdown("**Generated ADQL**")
    st.markdown(f'<div class="adql-box">{adql}</div>', unsafe_allow_html=True)

    # Tabs
    tab_sky, tab_chart, tab_data, tab_dl = st.tabs([
        "✦ Sky View", "◈ Chart", "⊞ Data", "↓ Export"
    ])

    with tab_sky:
        if "ra" in df.columns and "dec" in df.columns:
            fig = make_sky_plot(df, ra_c, dec_c, radius)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            buf = io.BytesIO()
            fig2 = make_sky_plot(df, ra_c, dec_c, radius)
            fig2.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="#03040a")
            plt.close(fig2)
            buf.seek(0)
            st.download_button("↓ Download Sky Plot", data=buf,
                               file_name="sky_plot.png", mime="image/png")
        else:
            st.info("Sky view requires RA and Dec columns.")

    with tab_chart:
        if intent == "color_histogram" and "bp_rp" in df.columns:
            fig = make_histogram_plot(df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif intent == "velocity_computation":
            fig = make_velocity_plot(df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif "phot_g_mean_mag" in df.columns and "bp_rp" in df.columns:
            # Colour-magnitude diagram
            fig, ax = plt.subplots(figsize=(8, 7), facecolor="#03040a")
            ax.set_facecolor("#03040a")
            mag  = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
            bprp = pd.to_numeric(df["bp_rp"],           errors="coerce")
            common = mag.dropna().index.intersection(bprp.dropna().index)
            if len(common) > 0:
                sc = ax.scatter(bprp[common], mag[common],
                                c=bprp[common], cmap="RdYlBu_r", s=3, alpha=0.7)
                ax.invert_yaxis()
                ax.set_xlabel("BP – RP (mag)", color="#556080", fontsize=8, fontfamily="monospace")
                ax.set_ylabel("G magnitude",   color="#556080", fontsize=8, fontfamily="monospace")
                ax.set_title("Colour–Magnitude Diagram", color="#c8d8f0",
                             fontsize=10, fontfamily="monospace")
                for spine in ax.spines.values(): spine.set_edgecolor("#1a2540")
                ax.tick_params(colors="#556080", labelsize=7)
                ax.grid(True, color="#0d1528", linewidth=0.5, alpha=0.6)
                cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
                cbar.set_label("BP–RP", color="#556080", fontsize=7, fontfamily="monospace")
                cbar.ax.yaxis.set_tick_params(color="#556080", labelsize=6, labelcolor="#556080")
                cbar.outline.set_edgecolor("#1a2540")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Add bp_rp and phot_g_mean_mag columns for a colour-magnitude diagram.")

    with tab_data:
        search = st.text_input("Filter rows:", placeholder="e.g. source_id value...")
        disp   = df
        if search:
            mask = df.astype(str).apply(lambda c: c.str.contains(search, case=False)).any(axis=1)
            disp = df[mask]
            st.caption(f"Showing {len(disp)} / {len(df)} rows")
        st.dataframe(disp, use_container_width=True, height=400)

    with tab_dl:
        csv = io.StringIO()
        df.to_csv(csv, index=False)
        st.download_button("↓ Download CSV",  data=csv.getvalue(),
                           file_name="gaia_results.csv", mime="text/csv",
                           use_container_width=True)
        st.download_button("↓ Download JSON", data=json.dumps(data["rows"], indent=2),
                           file_name="gaia_results.json", mime="application/json",
                           use_container_width=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:5rem 2rem;color:#1a2540;'>
        <div style='font-size:4rem;'>✦</div>
        <div style='font-family:"Space Mono",monospace;font-size:0.7rem;color:#556080;
                    letter-spacing:0.2em;text-transform:uppercase;margin-top:1rem;'>
            Enter a query in the sidebar and click Submit GPU Job
        </div>
    </div>""", unsafe_allow_html=True)