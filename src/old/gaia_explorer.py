import streamlit as st
from astroquery.gaia import Gaia
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Gaia Sky Explorer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state init ────────────────────────────────────────────────────────

for key, default in [
    ("result_df", pd.DataFrame()),
    ("extra_info", {}),
    ("has_result", False),
    ("result_q", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── questions metadata ────────────────────────────────────────────────────────

QUESTIONS = [
    (1,  "Stars in a direction",      "What stars are there in a certain direction of the sky?"),
    (2,  "Distance comparison",       "How far do we get toward the Galactic Center vs. North?"),
    (3,  "Star colors",               "What color are the stars?"),
    (4,  "Nearby star velocities",    "At what speed do stars closest to the Sun move?"),
    (5,  "High proper motion stars",  "How many 'High Proper Motion' stars are in our neighborhood?"),
    (6,  "White dwarfs",              "Can you find the White Dwarfs?"),
    (7,  "Disk escapers",             "Are there stars currently speeding away from the Galactic disk?"),
    (8,  "Precise measurements",      "Which stars have the most precise measurements?"),
    (9,  "Cluster brightness",        "What is the average brightness of stars in a specific cluster?"),
    (10, "Binary candidates",         "How many stars in a specific area have a wobble?"),
]

# ── sky vault visualization ───────────────────────────────────────────────────

def build_sky_vault(df: pd.DataFrame) -> go.Figure:
    df = df.dropna(subset=["ra", "dec"])
    if df.empty:
        return go.Figure()

    ra_rad  = np.radians(df["ra"].values)
    dec_rad = np.radians(df["dec"].values)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    fig = go.Figure()

    # celestial grid — meridians
    for r_deg in np.arange(0, 360, 30):
        r_rad   = np.radians(r_deg)
        d_range = np.linspace(-np.pi / 2, np.pi / 2, 50)
        fig.add_trace(go.Scatter3d(
            x=np.cos(d_range) * np.cos(r_rad),
            y=np.cos(d_range) * np.sin(r_rad),
            z=np.sin(d_range),
            mode="lines",
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

    # celestial grid — parallels
    for d_deg in np.arange(-60, 90, 30):
        d_rad   = np.radians(d_deg)
        r_range = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter3d(
            x=np.cos(d_rad) * np.cos(r_range),
            y=np.cos(d_rad) * np.sin(r_range),
            z=np.full_like(r_range, np.sin(d_rad)),
            mode="lines",
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

    # stars — colour by BP-RP if available
    has_color = "bp_rp" in df.columns and df["bp_rp"].notna().any()
    if has_color:
        marker = dict(
            size=3,
            color=df["bp_rp"],
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(title="BP-RP", thickness=12, len=0.6),
            opacity=0.9,
        )
    else:
        marker = dict(size=2, color="white", opacity=0.9)

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=marker,
        name="Stars",
        hovertemplate="RA: %{customdata[0]:.3f}°<br>Dec: %{customdata[1]:.3f}°<extra></extra>",
        customdata=np.stack([df["ra"], df["dec"]], axis=-1),
    ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            bgcolor="black",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=540,
        showlegend=False,
    )
    return fig

# ── cached gaia query functions ───────────────────────────────────────────────

@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q1(ra, dec, radius_deg):
    adql = f"""
    SELECT source_id, ra, dec, parallax, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND phot_g_mean_mag IS NOT NULL
    """
    r = Gaia.launch_job_async(adql).get_results()
    return r.to_pandas() if r and len(r) else pd.DataFrame()


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q2(direction, radius_deg, top):
    coords = {"center": (266.4, -28.9), "north": (192.85, 27.13)}
    ra, dec = coords[direction]
    adql = f"""
    SELECT TOP {top}
        source_id, ra, dec, parallax,
        1000.0 / parallax AS distance_pc
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND parallax > 0.1
    AND parallax_over_error > 5
    """
    r = Gaia.launch_job_async(adql).get_results()
    if not r or not len(r):
        return pd.DataFrame()
    return r.to_pandas().sort_values("parallax")


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q3(ra, dec, radius_deg):
    adql = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag,
        phot_bp_mean_mag, phot_rp_mean_mag,
        (phot_bp_mean_mag - phot_rp_mean_mag) AS bp_rp
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND phot_bp_mean_mag IS NOT NULL
    AND phot_rp_mean_mag IS NOT NULL
    AND phot_g_mean_mag < 19
    """
    r = Gaia.launch_job_async(adql).get_results()
    return r.to_pandas() if r and len(r) else pd.DataFrame()


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q4(min_parallax_mas, top):
    adql = f"""
    SELECT TOP {top}
        source_id, ra, dec, parallax,
        pmra, pmdec, radial_velocity,
        SQRT(pmra * pmra + pmdec * pmdec) AS pm_total_mas_yr,
        4.74 * SQRT(pmra * pmra + pmdec * pmdec) / parallax AS v_transverse_km_s
    FROM gaiadr3.gaia_source
    WHERE parallax >= {min_parallax_mas}
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND radial_velocity IS NOT NULL
    AND parallax_over_error > 10
    """
    r = Gaia.launch_job_async(adql).get_results()
    if not r or not len(r):
        return pd.DataFrame()
    return r.to_pandas().sort_values("parallax", ascending=False)


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q5(pm_threshold):
    count_adql = f"""
    SELECT COUNT(*) AS num_high_pm_stars
    FROM gaiadr3.gaia_source
    WHERE SQRT(pmra * pmra + pmdec * pmdec) > {pm_threshold}
    AND pmra IS NOT NULL AND pmdec IS NOT NULL
    """
    top_adql = f"""
    SELECT TOP 20
        source_id, ra, dec, parallax, pmra, pmdec,
        SQRT(pmra * pmra + pmdec * pmdec) AS pm_total_mas_yr
    FROM gaiadr3.gaia_source
    WHERE SQRT(pmra * pmra + pmdec * pmdec) > {pm_threshold}
    AND pmra IS NOT NULL AND pmdec IS NOT NULL
    AND phot_g_mean_mag < 19
    ORDER BY pm_total_mas_yr DESC
    """
    count_r = Gaia.launch_job_async(count_adql).get_results()
    top_r   = Gaia.launch_job_async(top_adql).get_results()
    count = int(count_r["num_high_pm_stars"][0]) if count_r and len(count_r) else 0
    df    = top_r.to_pandas() if top_r and len(top_r) else pd.DataFrame()
    return count, df


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q6(max_bp_rp, min_parallax_mas, min_g_mag, top):
    adql = f"""
    SELECT TOP {top}
        source_id, ra, dec, parallax,
        phot_g_mean_mag, bp_rp,
        1000.0 / parallax AS distance_pc
    FROM gaiadr3.gaia_source
    WHERE bp_rp < {max_bp_rp}
    AND bp_rp IS NOT NULL
    AND phot_g_mean_mag > {min_g_mag}
    AND parallax > {min_parallax_mas}
    AND parallax_over_error > 10
    """
    r = Gaia.launch_job_async(adql).get_results()
    if not r or not len(r):
        return pd.DataFrame()
    return r.to_pandas().sort_values("bp_rp")


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q7(min_lat, min_rv, top):
    adql = f"""
    SELECT TOP {top}
        source_id, ra, dec, l, b,
        radial_velocity, parallax, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE ABS(b) >= {min_lat}
    AND ABS(radial_velocity) >= {min_rv}
    AND radial_velocity IS NOT NULL
    AND phot_g_mean_mag IS NOT NULL
    ORDER BY ABS(radial_velocity) DESC
    """
    r = Gaia.launch_job_async(adql).get_results()
    return r.to_pandas() if r and len(r) else pd.DataFrame()


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q8(min_snr, mag_limit, top):
    adql = f"""
    SELECT TOP {top}
        source_id, ra, dec, parallax, parallax_error,
        parallax / parallax_error AS parallax_snr,
        phot_g_mean_mag,
        1000.0 / parallax AS distance_pc
    FROM gaiadr3.gaia_source
    WHERE parallax / parallax_error >= {min_snr}
    AND parallax > 0 AND parallax_error > 0
    AND phot_g_mean_mag < {mag_limit}
    """
    r = Gaia.launch_job_async(adql).get_results()
    if not r or not len(r):
        return pd.DataFrame()
    return r.to_pandas().sort_values("parallax_snr", ascending=False)


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q9(ra, dec, radius_deg, mag_limit):
    stats_adql = f"""
    SELECT COUNT(*) AS num_stars,
           AVG(phot_g_mean_mag) AS avg_g_mag,
           MIN(phot_g_mean_mag) AS brightest_g_mag,
           MAX(phot_g_mean_mag) AS faintest_g_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND phot_g_mean_mag IS NOT NULL
    AND phot_g_mean_mag < {mag_limit}
    """
    dist_adql = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND phot_g_mean_mag IS NOT NULL
    AND phot_g_mean_mag < {mag_limit}
    """
    stats_r = Gaia.launch_job_async(stats_adql).get_results()
    dist_r  = Gaia.launch_job_async(dist_adql).get_results()
    stats_df = stats_r.to_pandas() if stats_r and len(stats_r) else pd.DataFrame()
    dist_df  = dist_r.to_pandas()  if dist_r  and len(dist_r)  else pd.DataFrame()
    return stats_df, dist_df


@st.cache_data(show_spinner="Querying Gaia archive…")
def run_q10(ra, dec, radius_deg, min_noise, min_noise_sig, mag_limit, top):
    adql = f"""
    SELECT TOP {top}
        source_id, ra, dec, phot_g_mean_mag,
        astrometric_excess_noise,
        astrometric_excess_noise_sig,
        parallax, bp_rp
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND astrometric_excess_noise     > {min_noise}
    AND astrometric_excess_noise_sig > {min_noise_sig}
    AND phot_g_mean_mag              < {mag_limit}
    ORDER BY astrometric_excess_noise DESC
    """
    r = Gaia.launch_job_async(adql).get_results()
    return r.to_pandas() if r and len(r) else pd.DataFrame()

# ── layout ────────────────────────────────────────────────────────────────────

st.title("⭐ Gaia Sky Explorer")

left_col, right_col = st.columns([1, 2], gap="large")

# ── left column: question list + parameter inputs ─────────────────────────────

with left_col:
    st.subheader("10 Questions")
    selected_q = st.radio(
        "question",
        options=[q[0] for q in QUESTIONS],
        format_func=lambda n: f"**{n}.** {QUESTIONS[n - 1][2]}",
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Parameters")

    # ── per-question parameter widgets ────────────────────────────────────────

    if selected_q == 1:
        ra  = st.number_input("Right Ascension (RA °)", 0.0, 360.0, 56.75, step=0.5,
                              help="Pleiades default: 56.75")
        dec = st.number_input("Declination (Dec °)", -90.0, 90.0, 24.116, step=0.5,
                              help="Pleiades default: 24.116")
        rad = st.slider("Search radius (°)", 0.1, 5.0, 0.5, step=0.1)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q1(ra, dec, rad)
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 2:
        direction = st.selectbox(
            "Direction",
            ["center", "north"],
            format_func=lambda d: "Galactic Center" if d == "center" else "Galactic North Pole",
        )
        rad = st.slider("Search radius (°)", 0.5, 3.0, 1.0, step=0.5)
        top = st.number_input("Max rows", 100, 5000, 2000, step=100)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q2(direction, rad, int(top))
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 3:
        ra  = st.number_input("RA °", 0.0, 360.0, 56.75, step=0.5)
        dec = st.number_input("Dec °", -90.0, 90.0, 24.116, step=0.5)
        rad = st.slider("Search radius (°)", 0.1, 2.0, 0.5, step=0.1)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q3(ra, dec, rad)
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 4:
        min_par = st.number_input(
            "Min parallax (mas)", 10, 500, 50, step=10,
            help="100 mas ≈ within 10 pc of the Sun",
        )
        top = st.number_input("Max rows", 50, 2000, 500, step=50)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q4(int(min_par), int(top))
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 5:
        pm_thr = st.number_input("PM threshold (mas/yr)", 100, 5000, 500, step=100)
        run = st.button("Run Query", use_container_width=True)
        if run:
            count, df = run_q5(int(pm_thr))
            st.session_state.result_df  = df
            st.session_state.extra_info = {"count": count}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 6:
        max_bp_rp = st.number_input("Max BP-RP color", 0.1, 2.0, 0.6, step=0.1)
        min_par   = st.number_input("Min parallax (mas)", 5, 200, 20, step=5)
        min_g_mag = st.number_input("Min G magnitude", 5.0, 16.0, 12.0, step=0.5)
        top = st.number_input("Max rows", 50, 2000, 1000, step=50)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q6(max_bp_rp, float(min_par), float(min_g_mag), int(top))
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 7:
        min_lat = st.slider("Min |Galactic latitude| (°)", 10, 90, 60, step=5)
        min_rv  = st.number_input("Min |radial velocity| (km/s)", 0, 500, 100, step=10)
        top = st.number_input("Max rows", 50, 2000, 500, step=50)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q7(int(min_lat), int(min_rv), int(top))
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 8:
        min_snr   = st.number_input("Min parallax SNR", 10, 500, 50, step=10)
        mag_limit = st.number_input("G magnitude limit", 10.0, 20.0, 15.0, step=0.5)
        top = st.number_input("Max rows", 50, 2000, 1000, step=50)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q8(int(min_snr), float(mag_limit), int(top))
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 9:
        ra  = st.number_input("RA °", 0.0, 360.0, 56.75, step=0.5,
                              help="Pleiades: 56.75 / Hyades: 66.0")
        dec = st.number_input("Dec °", -90.0, 90.0, 24.116, step=0.5,
                              help="Pleiades: 24.116 / Hyades: 15.9")
        rad = st.slider("Search radius (°)", 0.1, 5.0, 0.5, step=0.1)
        mag_limit = st.number_input("G magnitude limit", 10.0, 22.0, 19.0, step=0.5)
        run = st.button("Run Query", use_container_width=True)
        if run:
            stats_df, dist_df = run_q9(ra, dec, rad, float(mag_limit))
            st.session_state.result_df  = dist_df
            st.session_state.extra_info = {"stats": stats_df}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

    elif selected_q == 10:
        ra  = st.number_input("RA °", 0.0, 360.0, 66.0, step=0.5,
                              help="Hyades default: RA=66.0, Dec=15.9")
        dec = st.number_input("Dec °", -90.0, 90.0, 15.9, step=0.5)
        rad = st.slider("Search radius (°)", 0.5, 10.0, 5.0, step=0.5)
        min_noise     = st.number_input("Min excess noise (mas)", 0.1, 10.0, 1.0, step=0.1)
        min_noise_sig = st.number_input("Min noise significance", 0.5, 10.0, 2.0, step=0.5)
        mag_limit     = st.number_input("G magnitude limit", 10.0, 22.0, 18.0, step=0.5)
        top = st.number_input("Max rows", 50, 2000, 1000, step=50)
        run = st.button("Run Query", use_container_width=True)
        if run:
            st.session_state.result_df  = run_q10(ra, dec, rad, float(min_noise),
                                                   float(min_noise_sig), float(mag_limit), int(top))
            st.session_state.extra_info = {}
            st.session_state.has_result = True
            st.session_state.result_q   = selected_q

# ── right column: sky vault + dataset + download ──────────────────────────────

with right_col:
    q_num, q_short, q_full = QUESTIONS[selected_q - 1]
    st.markdown(f"### Q{q_num}. {q_short}")
    st.caption(q_full)

    if not st.session_state.has_result:
        st.info("Set parameters on the left and press **Run Query** to query the Gaia archive.")

    elif st.session_state.result_df.empty:
        st.warning("No results returned. Try relaxing your parameters.")

    else:
        df = st.session_state.result_df

        # ── extra metrics (Q5 count, Q9 stats) ───────────────────────────────
        info = st.session_state.extra_info

        if "count" in info:
            st.metric(
                f"Total stars with PM > threshold in the full catalogue",
                f"{info['count']:,}",
            )
            st.caption("Table below shows the top 20 fastest movers.")

        if "stats" in info and not info["stats"].empty:
            s = info["stats"].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stars in region", f"{int(s['num_stars']):,}")
            c2.metric("Avg G mag",       f"{float(s['avg_g_mag']):.2f}")
            c3.metric("Brightest",       f"{float(s['brightest_g_mag']):.2f}")
            c4.metric("Faintest",        f"{float(s['faintest_g_mag']):.2f}")

        # ── sky vault ─────────────────────────────────────────────────────────
        if "ra" in df.columns and "dec" in df.columns:
            st.plotly_chart(build_sky_vault(df), use_container_width=True)
        else:
            st.info("No sky position columns (ra/dec) in this result — vault not available.")

        # ── dataset table + download ──────────────────────────────────────────
        st.markdown(f"**Dataset — {len(df):,} rows × {len(df.columns)} columns**")
        st.dataframe(df, use_container_width=True, height=300)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        fname     = f"gaia_q{st.session_state.result_q}_results.csv"
        st.download_button(
            label="⬇️  Download as CSV",
            data=csv_bytes,
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )
