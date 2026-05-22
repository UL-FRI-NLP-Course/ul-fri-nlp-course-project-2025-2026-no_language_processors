import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from IPython.display import display, HTML
from datetime import datetime

# ── Greyscale palette ─────────────────────────────────────────────────────────
G0  = '#111111'   # near black  — headings, borders
G1  = '#333333'   # dark grey   — body text, titles
G2  = '#555555'   # mid grey    — labels, subtitles
G3  = '#888888'   # light grey  — muted text
G4  = '#cccccc'   # pale grey   — borders, rules
G5  = '#e8e8e8'   # very pale   — surfaces, hover
G6  = '#f4f4f4'   # near white  — backgrounds

# ── Matplotlib style ──────────────────────────────────────────────────────────
MPL_STYLE = {
    'figure.facecolor':  'white',
    'axes.facecolor':    '#fafafa',
    'axes.edgecolor':    G4,
    'axes.labelcolor':   G1,
    'axes.titlecolor':   G0,
    'xtick.color':       G2,
    'ytick.color':       G2,
    'text.color':        G1,
    'grid.color':        G4,
    'grid.alpha':        0.6,
    'grid.linestyle':    '--',
    'axes.grid':         True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         11,
}

GREY_CMAP    = LinearSegmentedColormap.from_list('greys',  ['#111111', '#666666', '#bbbbbb'], N=256)
SKY_CMAP     = LinearSegmentedColormap.from_list('sky',    ['#0b3d91', '#2166ac', '#74add1', '#ffffbf'], N=256)
HR_CMAP      = LinearSegmentedColormap.from_list('hr',     ['#2166ac', '#92c5de', '#fddbc7', '#d6604d', '#b2182b'], N=256)
PM_CMAP      = LinearSegmentedColormap.from_list('pm',     ['#edf8fb', '#4eb3d3', '#08519c'], N=256)
MAG_CMAP     = LinearSegmentedColormap.from_list('mag',    ['#08306b', '#2171b5', '#6baed6', '#c6dbef'], N=256)
COLOUR_CMAP  = LinearSegmentedColormap.from_list('colour', ['#2166ac', '#92c5de', '#f7f7f7', '#d6604d', '#b2182b'], N=256)
SPHERE_CMAP  = LinearSegmentedColormap.from_list('sphere', ['#ffffff', '#ffe4b5', '#ffaa00', '#cc3300'], N=256)


# ── Helper: fig → base64 PNG ──────────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Flat RA/Dec sky scatter ───────────────────────────────────────────────────
def _build_sky_scatter(df: pd.DataFrame) -> str:
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colour_col = 'phot_g_mean_mag' if 'phot_g_mean_mag' in df.columns else None
        if colour_col:
            sc = ax.scatter(df['ra'], df['dec'], c=df[colour_col],
                            cmap=SPHERE_CMAP, s=10, alpha=0.80,
                            linewidths=0, rasterized=True)
            cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
            cb.set_label('G magnitude', fontsize=11, color='#08306b')
            cb.ax.tick_params(labelsize=10, colors=G2)
        else:
            ax.scatter(df['ra'], df['dec'], c='#2166ac', s=8,
                       alpha=0.65, linewidths=0, rasterized=True)
        ax.invert_xaxis()
        ax.set_xlabel('Right Ascension (deg)', fontsize=12)
        ax.set_ylabel('Declination (deg)', fontsize=12)
        ax.set_title(f'Sky Distribution  —  {len(df):,} sources',
                     fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        b64 = _fig_to_b64(fig)
        plt.close(fig)
    return b64


# ── Colour histogram (BP-RP) ──────────────────────────────────────────────────
def _build_colour_histogram(df: pd.DataFrame):
    if 'bp_rp' not in df.columns:
        return None
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        bprp   = df['bp_rp'].dropna()
        n_bins = min(40, max(10, len(bprp) // 4))
        counts, bins, patches = ax.hist(bprp, bins=n_bins,
                                        edgecolor='white', linewidth=0.5)
        norm = plt.Normalize(bins.min(), bins.max())
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(COLOUR_CMAP(norm(left)))
        ax.set_xlabel('BP - RP colour index', fontsize=12)
        ax.set_ylabel('Number of stars', fontsize=12)
        ax.set_title('Stellar Colour Distribution', fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=11)
        ax.text(0.02, 0.91, 'hot / blue stars', transform=ax.transAxes,
                fontsize=10, color='#2166ac')
        ax.text(0.72, 0.91, 'cool / red stars', transform=ax.transAxes,
                fontsize=10, color='#b2182b')
        plt.tight_layout()
        b64 = _fig_to_b64(fig)
        plt.close(fig)
    return b64


# ── Supplementary: magnitude + proper motion + HR ─────────────────────────────
def _build_supplementary_plots(df: pd.DataFrame):
    has_mag = 'phot_g_mean_mag' in df.columns
    has_pm  = 'pmra' in df.columns and 'pmdec' in df.columns
    has_hr  = 'bp_rp' in df.columns and 'phot_g_mean_mag' in df.columns
    panels  = [x for x in [has_mag, has_pm, has_hr] if x]
    if not panels:
        return None

    n = len(panels)
    with plt.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.5))
        if n == 1:
            axes = [axes]
        idx = 0

        if has_mag:
            ax  = axes[idx]; idx += 1
            mag = df['phot_g_mean_mag'].dropna()
            nb  = min(30, max(10, len(mag) // 5))
            _, bins, patches = ax.hist(mag, bins=nb,
                                       edgecolor='white', linewidth=0.5)
            norm = plt.Normalize(bins.min(), bins.max())
            for patch, left in zip(patches, bins[:-1]):
                patch.set_facecolor(MAG_CMAP(norm(left)))
            ax.axvline(mag.mean(), color='#08306b', lw=1.8, ls='--',
                       label=f'mean = {mag.mean():.1f}', alpha=0.85)
            ax.set_xlabel('G magnitude', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Magnitude Distribution', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, framealpha=0.7)
            ax.tick_params(labelsize=11)

        if has_pm:
            ax    = axes[idx]; idx += 1
            pm    = df[['pmra', 'pmdec']].dropna()
            speed = np.sqrt(pm['pmra']**2 + pm['pmdec']**2)
            sc = ax.scatter(pm['pmra'], pm['pmdec'], c=speed,
                            cmap=PM_CMAP, s=10, alpha=0.75, linewidths=0)
            cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
            cb.set_label('|pm| (mas/yr)', fontsize=11, color='#08306b')
            cb.ax.tick_params(labelsize=10, colors=G2)
            ax.axhline(0, color=G4, lw=1.2)
            ax.axvline(0, color=G4, lw=1.2)
            ax.set_xlabel('pmRA (mas/yr)', fontsize=12)
            ax.set_ylabel('pmDec (mas/yr)', fontsize=12)
            ax.set_title('Proper Motion', fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=11)

        if has_hr:
            ax = axes[idx]; idx += 1
            hr = df[['bp_rp', 'phot_g_mean_mag']].dropna()
            norm3 = plt.Normalize(hr['bp_rp'].quantile(0.02),
                                  hr['bp_rp'].quantile(0.98))
            sc3 = ax.scatter(hr['bp_rp'], hr['phot_g_mean_mag'],
                             c=hr['bp_rp'], cmap=HR_CMAP, norm=norm3,
                             s=6, alpha=0.70, linewidths=0)
            cb3 = plt.colorbar(sc3, ax=ax, pad=0.02, fraction=0.03)
            cb3.set_label('BP - RP', fontsize=11, color='#08306b')
            cb3.ax.tick_params(labelsize=10, colors=G2)
            ax.invert_yaxis()
            ax.set_xlabel('BP - RP (colour)', fontsize=12)
            ax.set_ylabel('G magnitude', fontsize=12)
            ax.set_title('Colour-Magnitude Diagram', fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=11)

        plt.tight_layout(pad=1.8)
        b64 = _fig_to_b64(fig)
        plt.close(fig)
    return b64


# ── Plotly celestial sphere ───────────────────────────────────────────────────
def _build_sphere_html(df: pd.DataFrame) -> str:
    ra    = df['ra'].to_numpy(float)
    dec   = df['dec'].to_numpy(float)
    ra_r  = np.radians(ra)
    dec_r = np.radians(dec)
    x = np.cos(dec_r) * np.cos(ra_r)
    y = np.cos(dec_r) * np.sin(ra_r)
    z = np.sin(dec_r)

    colour_col = 'phot_g_mean_mag' if 'phot_g_mean_mag' in df.columns else None
    if colour_col:
        cv = df[colour_col].to_numpy(float)
        mk = dict(
            color=cv,
            colorscale=[[0.0, '#ffffff'], [0.35, '#ffe4b5'],
                        [0.65, '#ffaa00'], [1.0, '#cc3300']],
            reversescale=False,
            colorbar=dict(
                title=dict(text='G mag', font=dict(color='#1a3a5c', size=12)),
                thickness=12, len=0.5, x=1.01,
                tickfont=dict(color=G2, size=11),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
            ),
            showscale=True, size=3.5, opacity=1.0,
        )
        custom = np.stack([ra, dec, cv], axis=1)
        hover  = ('RA: %{customdata[0]:.3f}<br>'
                  'Dec: %{customdata[1]:.3f}<br>'
                  'G: %{customdata[2]:.2f}<extra></extra>')
    else:
        mk     = dict(color='#2166ac', showscale=False, size=2.5, opacity=0.9)
        custom = np.stack([ra, dec], axis=1)
        hover  = 'RA: %{customdata[0]:.3f}<br>Dec: %{customdata[1]:.3f}<extra></extra>'

    stars = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                         marker=mk, hovertemplate=hover,
                         customdata=custom, name='Stars')

    # Grid lines — dark and wide enough to be clearly visible
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(-np.pi/2, np.pi/2, 18)
    lx, ly, lz = [], [], []
    for vi in v[::3]:
        lx += [*np.cos(vi)*np.cos(u), None]
        ly += [*np.cos(vi)*np.sin(u), None]
        lz += [*np.full_like(u, np.sin(vi)), None]
    for ui in u[::3]:
        lx += [*np.cos(v)*np.cos(ui), None]
        ly += [*np.cos(v)*np.sin(ui), None]
        lz += [*np.sin(v), None]
    grid = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                        line=dict(color='rgba(220,230,255,0.55)', width=3),
                        hoverinfo='skip', showlegend=False)

    # Equator — bold black line
    eq = np.linspace(0, 2*np.pi, 200)
    equator = go.Scatter3d(x=np.cos(eq), y=np.sin(eq), z=np.zeros(200),
                           mode='lines',
                            line=dict(color='rgba(240,245,255,0.90)', width=6),
                           hoverinfo='skip', showlegend=False)

    ax_s = dict(showbackground=False, showgrid=False,
                zeroline=False, showticklabels=False, title='')
    layout = go.Layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0d1b3e',
        scene=dict(xaxis=ax_s, yaxis=ax_s, zaxis=ax_s,
                   bgcolor='#0d1b3e',
                   camera=dict(eye=dict(x=1.4, y=1.4, z=0.6))),
        legend=dict(font=dict(color=G2), bgcolor='rgba(0,0,0,0)'),
    )
    fig = go.Figure(data=[grid, equator, stars], layout=layout)
    return fig.to_html(full_html=False, include_plotlyjs='cdn',
                       config={'responsive': True, 'displayModeBar': False})


# ── Summary stats ─────────────────────────────────────────────────────────────
def _summary_stats(df: pd.DataFrame, query: str) -> dict:
    num = df.select_dtypes(include='number')
    return {
        'n_rows':    len(df),
        'n_cols':    len(df.columns),
        'query':     query,
        'timestamp': datetime.now().strftime('%d %B %Y, %H:%M UTC'),
        'numeric_summary': {
            col: {
                'min':  f'{num[col].min():.5g}',
                'max':  f'{num[col].max():.5g}',
                'mean': f'{num[col].mean():.5g}',
                'null': int(num[col].isnull().sum()),
            }
            for col in num.columns
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, user_query: str,
                    output_path: str = 'gaia_report.html'):
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print('No ra/dec columns — cannot build report.')
        return

    print('Building report ...')
    stats      = _summary_stats(df, user_query)
    sphere_div = _build_sphere_html(df)
    sky_b64    = _build_sky_scatter(df)
    supp_b64   = _build_supplementary_plots(df)
    colour_b64 = _build_colour_histogram(df)

    # Table (first 200 rows)
    preview = df.head(200).copy()
    for col in preview.select_dtypes(include='float').columns:
        preview[col] = preview[col].round(5)
    col_headers = ''.join(f'<th>{c}</th>' for c in preview.columns)
    table_rows  = ''.join(
        '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
        for _, row in preview.iterrows()
    )

    # Stat cards
    stat_cards = ''
    for col, s in stats['numeric_summary'].items():
        warn = ' warn' if s['null'] > 0 else ''
        stat_cards += f"""
        <div class="stat-card">
          <div class="stat-col">{col}</div>
          <div class="stat-row"><span class="lbl">min</span><span class="val">{s['min']}</span></div>
          <div class="stat-row"><span class="lbl">max</span><span class="val">{s['max']}</span></div>
          <div class="stat-row"><span class="lbl">mean</span><span class="val">{s['mean']}</span></div>
          <div class="stat-row"><span class="lbl">nulls</span><span class="val{warn}">{s['null']}</span></div>
        </div>"""

    supp_section = ''
    if supp_b64:
        supp_section = f"""
        <section class="plot-section">
          <h2 class="section-title">Statistical Plots</h2>
          <div class="plot-box">
            <img src="data:image/png;base64,{supp_b64}" alt="Statistical plots"/>
          </div>
        </section>"""

    colour_right = (
        f'<div class="plot-box">'
        f'<img src="data:image/png;base64,{colour_b64}" alt="Colour histogram"/>'
        f'</div>'
        if colour_b64
        else '<div class="plot-box no-data">No BP-RP colour data available</div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Gaia Intelligent Query Pipeline</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --g0:      #111111;
    --g1:      #333333;
    --g2:      #555555;
    --g3:      #888888;
    --g4:      #cccccc;
    --g5:      #e8e8e8;
    --g6:      #f4f4f4;
    --bg:      #ffffff;
    --title:   #1a3a5c;
    --radius:  5px;
    --font:    Arial, Helvetica, sans-serif;
    --mono:    'Courier New', Courier, monospace;
  }}

  html, body {{
    background: var(--bg);
    color: var(--g1);
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.6;
  }}

  .page {{
    max-width: 1560px;
    margin: 0 auto;
    padding: 40px 36px 60px;
  }}

  /* ── Header ── */
  header {{
    border-bottom: 3px solid var(--title);
    padding-bottom: 26px;
    margin-bottom: 36px;
  }}

  .title-row {{
    display: flex;
    align-items: baseline;
    gap: 18px;
    margin-bottom: 20px;
  }}
  h1 {{
    font-family: var(--font);
    font-size: 28px;
    font-weight: 700;
    color: var(--title);
    letter-spacing: -0.3px;
  }}
  .subtitle {{
    font-size: 14px;
    color: var(--g3);
    font-weight: 400;
  }}

  .question-block {{
    background: var(--g6);
    border-left: 4px solid var(--title);
    padding: 14px 20px;
    border-radius: 0 var(--radius) var(--radius) 0;
    margin-bottom: 16px;
  }}
  .question-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--title);
    margin-bottom: 6px;
  }}
  .question-text {{
    font-size: 18px;
    font-weight: 600;
    color: var(--title);
    line-height: 1.4;
  }}

  .meta-row {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
  }}
  .pill {{
    border: 1px solid var(--g4);
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: var(--mono);
    color: var(--g2);
    background: var(--g6);
  }}
  .pill b {{ color: var(--title); }}

  /* ── Section titles ── */
  .section-title {{
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--title);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--g4);
  }}

  /* ── Main two-column ── */
  .main-panel {{
    display: grid;
    grid-template-columns: 1fr 1.18fr;
    gap: 22px;
    margin-bottom: 32px;
    align-items: start;
  }}

  /* ── Table ── */
  .table-panel {{
    border: 1px solid var(--g4);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .panel-header {{
    background: var(--g6);
    border-bottom: 1px solid var(--g4);
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .ptitle {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--title);
  }}
  .pcount {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--g3);
  }}
  .table-scroll {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: 480px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: var(--mono);
    white-space: nowrap;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: var(--g5);
    color: var(--title);
    font-family: var(--font);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    padding: 8px 12px;
    text-align: right;
    border-bottom: 1px solid var(--g4);
  }}
  thead th:first-child {{ text-align: left; }}
  tbody tr {{
    border-bottom: 1px solid var(--g5);
    transition: background 0.1s;
  }}
  tbody tr:nth-child(even) {{ background: var(--g6); }}
  tbody tr:hover {{ background: var(--g5); }}
  tbody td {{
    padding: 5px 12px;
    text-align: right;
    color: var(--g1);
  }}
  tbody td:first-child {{
    text-align: left;
    color: var(--g2);
  }}

  /* ── Sphere ── */
  .sphere-panel {{
    border: 1px solid var(--g4);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .sphere-inner {{
    min-height: 480px;
    background: var(--bg);
  }}
  .sphere-inner > div {{
    height: 100% !important;
    min-height: 480px;
  }}

  /* ── Stat cards ── */
  .stats-section {{ margin-bottom: 32px; }}
  .stats-grid {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .stat-card {{
    background: var(--g6);
    border: 1px solid var(--g4);
    border-radius: var(--radius);
    padding: 12px 14px;
    min-width: 130px;
    flex: 1;
  }}
  .stat-col {{
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    color: var(--title);
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--g4);
    padding-bottom: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
  }}
  .lbl {{ font-size: 11px; color: var(--g3); }}
  .val {{ font-family: var(--mono); font-size: 11px; color: var(--g1); }}
  .val.warn {{ color: #aa3300; font-weight: 700; }}

  /* ── Plots ── */
  .plot-section {{ margin-bottom: 32px; }}
  .plot-box {{
    border: 1px solid var(--g4);
    border-radius: var(--radius);
    background: var(--g6);
    padding: 16px;
    text-align: center;
  }}
  .plot-box img {{
    max-width: 100%;
    border-radius: 3px;
    display: block;
    margin: 0 auto;
  }}
  .plot-box.no-data {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    color: var(--g3);
    font-size: 14px;
  }}
  .plot-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
    margin-bottom: 32px;
  }}

  /* ── Footer ── */
  footer {{
    border-top: 1px solid var(--g4);
    padding-top: 18px;
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 12px;
    color: var(--g3);
  }}
  footer b {{ color: var(--title); font-weight: 600; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: var(--g6); }}
  ::-webkit-scrollbar-thumb {{ background: var(--g4); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--g2); }}
</style>
</head>
<body>
<div class="page">

  <header>
    <div class="title-row">
      <h1>Gaia Intelligent Query Pipeline</h1>
      <span class="subtitle">ESA Gaia Data Release 3 — Query Summary</span>
    </div>
    <div class="question-block">
      <div class="question-label">Question</div>
      <div class="question-text">{stats['query']}</div>
    </div>
    <div class="meta-row">
      <div class="pill"><b>{stats['n_rows']:,}</b> sources retrieved</div>
      <div class="pill"><b>{stats['n_cols']}</b> columns</div>
      <div class="pill">gaiadr3.gaia_source</div>
      <div class="pill">{stats['timestamp']}</div>
    </div>
  </header>

  <h2 class="section-title">Data &amp; Sky Projection</h2>
  <div class="main-panel">

    <div class="table-panel">
      <div class="panel-header">
        <span class="ptitle">Retrieved Dataset</span>
        <span class="pcount">{min(200, len(df)):,} / {len(df):,} rows shown</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr>{col_headers}</tr></thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="sphere-panel">
      <div class="panel-header">
        <span class="ptitle">Celestial Sphere</span>
        <span class="pcount">drag · zoom · rotate</span>
      </div>
      <div class="sphere-inner">{sphere_div}</div>
    </div>

  </div>

  <section class="stats-section">
    <h2 class="section-title">Column Statistics</h2>
    <div class="stats-grid">{stat_cards}</div>
  </section>

  <h2 class="section-title">Sky Map &amp; Colour Analysis</h2>
  <div class="plot-row">
    <div class="plot-box">
      <img src="data:image/png;base64,{sky_b64}" alt="Sky scatter"/>
    </div>
    {colour_right}
  </div>

  {supp_section}

  <footer>
    <div>Gaia Data Release 3 — European Space Agency — via <b>astroquery</b> TAP</div>
    <div>Generated by <b>Gaia Intelligent Query Pipeline</b> — {stats['timestamp']}</div>
  </footer>

</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = len(html.encode()) / 1024
    print(f'Report saved: {output_path}  ({size_kb:.0f} KB)')

    display(HTML(f'''
    <div style="font-family:Arial,sans-serif; background:#f4f4f4;
                border:1px solid #cccccc; border-left:4px solid #111111;
                padding:11px 16px; border-radius:4px; margin-top:8px; color:#111111;">
      Report saved:
      <a href="{output_path}" target="_blank"
         style="color:#333333; font-weight:bold; text-decoration:underline;">{output_path}</a>
      <span style="color:#888888; font-size:12px; margin-left:8px;">({size_kb:.0f} KB)</span>
    </div>
    '''))


if __name__ == '__main__': 
    generate_report(df, USER_QUERY, output_path='gaia_report.html')