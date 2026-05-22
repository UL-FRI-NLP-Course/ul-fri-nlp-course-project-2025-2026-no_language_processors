"""
parser.py — Gaia DR3 schema constants, LLM query parser, and ADQL builder.
"""

from config import json, re, pd
from model  import llm, SAMPLING_PARAMS

# ── Valid Gaia DR3 columns ────────────────────────────────────────────────────
VALID_COLUMNS = {
    'ra', 'dec', 'source_id', 'parallax', 'parallax_error',
    'pmra', 'pmdec', 'radial_velocity',
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp',
    'teff_gspphot', 'logg_gspphot', 'mh_gspphot',
    'phot_variable_flag', 'ruwe',
}

VALID_JOIN_TABLES = {
    'astrophysical_parameters',
    'vari_classifier_result',
    'vari_rrlyrae',
    'vari_cepheid',
    'vari_eclipsing_binary',
    'vari_rotation_modulation',
}

VALID_INTENTS = {
    'cone_search',
    'color_histogram',
    'hr_diagram',
    'stellar_population',
    'variability_search',
    'internal_crossmatch',
    'nearest_neighbours',
    'velocity_computation',
}

_COLUMN_ALIASES = {
    'magnitude':         'phot_g_mean_mag',
    'g_magnitude':       'phot_g_mean_mag',
    'g_mag':             'phot_g_mean_mag',
    'bp_magnitude':      'phot_bp_mean_mag',
    'rp_magnitude':      'phot_rp_mean_mag',
    'color':             'bp_rp',
    'colour':            'bp_rp',
    'temperature':       'teff_gspphot',
    'teff':              'teff_gspphot',
    'logg':              'logg_gspphot',
    'metallicity':       'mh_gspphot',
    'proper_motion_ra':  'pmra',
    'proper_motion_dec': 'pmdec',
    'variability':       'phot_variable_flag',
}

SYSTEM_PROMPT = """You are an astronomy query parser for the Gaia DR3 database.

Convert the user query into a single JSON object. Return ONLY valid JSON — no markdown, no explanation.

JSON schema:
{
  "intent":     string,
  "ra":         float or null,
  "dec":        float or null,
  "radius":     float or null,
  "columns":    [list of strings],
  "filters":    {dict of column: value or {min, max}},
  "join_table": string or null,
  "limit":      int
}

Supported intents:
  cone_search          — stars in a circular sky region (ra, dec, radius required)
  color_histogram      — colour distribution in a region (ra, dec, radius required)
  hr_diagram           — HR diagram; cone optional for sky-wide (parallax+G+bp_rp always added)
  stellar_population   — filter by star type; cone optional for sky-wide queries
  variability_search   — variable stars in a region (ra, dec, radius required)
  internal_crossmatch  — join with another DR3 table; cone optional (join_table required)
  nearest_neighbours   — closest stars to the Sun, sky-wide (ra/dec/radius may be null)
  velocity_computation — 3D kinematics: proper motion + radial velocity (ra/dec/radius optional)

Valid column names:
  ra, dec, source_id, parallax, parallax_error,
  pmra, pmdec, radial_velocity,
  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
  teff_gspphot, logg_gspphot, mh_gspphot,
  phot_variable_flag, ruwe

Valid join_table values (internal_crossmatch only):
  astrophysical_parameters, vari_classifier_result, vari_rrlyrae,
  vari_cepheid, vari_eclipsing_binary, vari_rotation_modulation

Rules:
  - If a target name is given (e.g. 'Pleiades', 'Andromeda'), infer ra/dec from your knowledge.
  - If coordinates are unknown and intent requires them, use galactic centre: ra=266.4, dec=-29.0
  - Default radius: 1.0 degree
  - Default limit: 1000
  - ra/dec/radius may be null for nearest_neighbours, velocity_computation, hr_diagram,
    stellar_population, internal_crossmatch
  - ra/dec/radius are REQUIRED for cone_search, color_histogram, variability_search
  - join_table is null unless intent is internal_crossmatch
  - filters: use {"min": x, "max": y} for ranges, or a single value for equality
  - "brightest", "faintest", "hottest", "coolest", "fastest", "nearest" —
    these are sky-wide superlatives, never use cone_search for them.

Examples:

Query: "Show stars around the Pleiades"
{"intent":"cone_search","ra":56.75,"dec":24.12,"radius":1.0,
 "columns":["source_id","ra","dec","phot_g_mean_mag"],
 "filters":{},"join_table":null,"limit":1000}

Query: "HR diagram of stars near Omega Centauri"
{"intent":"hr_diagram","ra":201.7,"dec":-47.5,"radius":1.0,
 "columns":["source_id","parallax","phot_g_mean_mag","bp_rp"],
 "filters":{"parallax":{"min":0.1}},"join_table":null,"limit":2000}

Query: "Show me the 50 nearest stars"
{"intent":"nearest_neighbours","ra":null,"dec":null,"radius":null,
 "columns":["source_id","ra","dec","parallax","phot_g_mean_mag"],
 "filters":{},"join_table":null,"limit":50}

Query: "Red dwarfs near Barnard's star"
{"intent":"stellar_population","ra":269.45,"dec":4.69,"radius":2.0,
 "columns":["source_id","ra","dec","teff_gspphot","logg_gspphot","bp_rp"],
 "filters":{"teff_gspphot":{"min":2400,"max":3900},"logg_gspphot":{"min":4.5}},
 "join_table":null,"limit":1000}

Query: "Variable stars near the galactic centre"
{"intent":"variability_search","ra":266.4,"dec":-29.0,"radius":1.0,
 "columns":["source_id","ra","dec","phot_g_mean_mag","phot_variable_flag"],
 "filters":{"phot_variable_flag":"VARIABLE"},"join_table":null,"limit":1000}

Query: "How fast do stars near Orion move?"
{"intent":"velocity_computation","ra":83.8,"dec":-5.4,"radius":2.0,
 "columns":["source_id","ra","dec","pmra","pmdec","radial_velocity","parallax"],
 "filters":{"radial_velocity":{"min":-500,"max":500}},"join_table":null,"limit":1000}

Query: "Colour histogram of stars near Andromeda"
{"intent":"color_histogram","ra":10.68,"dec":41.27,"radius":0.5,
 "columns":["source_id","bp_rp","phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag"],
 "filters":{"bp_rp":{"min":-1.0,"max":6.0}},"join_table":null,"limit":1000}

Query: "Show me the brightest stars in the sky"
{"intent":"stellar_population","ra":null,"dec":null,"radius":null,
 "columns":["source_id","ra","dec","phot_g_mean_mag","bp_rp","parallax"],
 "filters":{"phot_g_mean_mag":{"max":4.0}},
 "join_table":null,"limit":1000}

Query: "Cross-match stars near the LMC with astrophysical parameters"
{"intent":"internal_crossmatch","ra":80.9,"dec":-69.8,"radius":2.0,
 "columns":["source_id","ra","dec","teff_gspphot","logg_gspphot"],
 "filters":{},"join_table":"astrophysical_parameters","limit":1000}
"""


# ── ADQL helper functions ─────────────────────────────────────────────────────

def _resolve_columns(cols: list) -> list:
    resolved = [_COLUMN_ALIASES.get(c, c) for c in cols]
    seen, out = set(), []
    for c in resolved:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _cone_clause(ra, dec, radius, ra_col='ra', dec_col='dec') -> str:
    return (
        f"1=CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )


def _filter_clauses(filters: dict) -> list:
    clauses = []
    for col, val in filters.items():
        col = _COLUMN_ALIASES.get(col, col)
        if isinstance(val, dict):
            mn, mx = val.get('min'), val.get('max')
            if mn is not None and mx is not None:
                clauses.append(f'{col} BETWEEN {mn} AND {mx}')
            elif mn is not None:
                clauses.append(f'{col} >= {mn}')
            elif mx is not None:
                clauses.append(f'{col} <= {mx}')
        elif isinstance(val, str):
            clauses.append(f"{col} = '{val}'")
        else:
            clauses.append(f'{col} = {val}')
    return clauses


def _null_guards(columns: list) -> list:
    gspphot = {'teff_gspphot', 'logg_gspphot', 'mh_gspphot'}
    return [f'{c} IS NOT NULL' for c in columns if c in gspphot]


# ── LLM parser ────────────────────────────────────────────────────────────────

def _build_prompt(user_query: str) -> str:
    return (
        '<|im_start|>system\n'
        f'{SYSTEM_PROMPT}<|im_end|>\n'
        '<|im_start|>user\n'
        f'{user_query}<|im_end|>\n'
        '<|im_start|>assistant\n'
    )


def parse_query(user_query: str) -> dict:
    """Send query to LLM and return parsed JSON dict."""
    prompt  = _build_prompt(user_query)
    outputs = llm.generate([prompt], SAMPLING_PARAMS)
    raw     = outputs[0].outputs[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'^```\s*',     '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$',     '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Model returned non-JSON:\n{raw}') from exc


# ── ADQL builder ──────────────────────────────────────────────────────────────

def build_adql(q: dict, source_id_filter: list = None) -> str:
    """
    Convert a validated parsed JSON dict into an ADQL query string.
    Optionally injects a source_id IN (...) filter for sequential steps.
    """
    intent     = q['intent']
    ra         = q.get('ra')
    dec        = q.get('dec')
    radius     = q.get('radius', 1.0)
    limit      = q.get('limit', 1000)
    filters    = q.get('filters') or {}
    join_table = q.get('join_table')
    columns    = _resolve_columns(q.get('columns') or ['source_id', 'ra', 'dec'])

    for col in ('source_id', 'ra', 'dec', 'phot_g_mean_mag', 'bp_rp', 'pmra', 'pmdec'):
        if col not in columns:
            columns.append(col)

    sky_wide     = ra is None or dec is None
    filter_parts = _filter_clauses(filters)

    if intent == 'cone_search':
        cols  = ', '.join(columns)
        where = ' AND '.join([_cone_clause(ra, dec, radius)] + filter_parts)
        adql  = f'SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {where} ORDER BY random_index'

    elif intent == 'color_histogram':
        forced = ['source_id', 'bp_rp', 'phot_g_mean_mag',
                  'phot_bp_mean_mag', 'phot_rp_mean_mag']
        cols   = ', '.join(_resolve_columns(forced + columns))
        where  = ' AND '.join([_cone_clause(ra, dec, radius), 'bp_rp IS NOT NULL'] + filter_parts)
        adql   = f'SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {where} ORDER BY random_index'

    elif intent == 'hr_diagram':
        forced      = ['source_id', 'parallax', 'phot_g_mean_mag', 'bp_rp']
        cols        = _resolve_columns(forced + columns)
        select_cols = ', '.join(cols)
        select_cols += ', (phot_g_mean_mag + 5*LOG10(parallax/100)) AS abs_g_mag'
        where_parts = ['parallax > 0', 'bp_rp IS NOT NULL'] + filter_parts
        if not sky_wide:
            where_parts.insert(0, _cone_clause(ra, dec, radius))
        adql = (f'SELECT TOP {limit} {select_cols} FROM gaiadr3.gaia_source '
                f'WHERE {" AND ".join(where_parts)} ORDER BY random_index')

    elif intent == 'stellar_population':
        cols        = ', '.join(columns)
        null_guards = _null_guards(columns)
        where_parts = null_guards + filter_parts
        if not sky_wide:
            where_parts.insert(0, _cone_clause(ra, dec, radius))
        where = ' AND '.join(where_parts) if where_parts else '1=1'
        adql  = f'SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {where} ORDER BY random_index'

    elif intent == 'variability_search':
        if 'phot_variable_flag' not in columns:
            columns.append('phot_variable_flag')
        cols        = ', '.join(columns)
        where_parts = [
            _cone_clause(ra, dec, radius),
            "phot_variable_flag = 'VARIABLE'",
        ] + [fp for fp in filter_parts if 'phot_variable_flag' not in fp]
        adql = (f'SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source '
                f'WHERE {" AND ".join(where_parts)} ORDER BY random_index')

    elif intent == 'internal_crossmatch':
        g_cols      = ', '.join(f'g.{c}' if c in VALID_COLUMNS else c for c in columns)
        where_parts = filter_parts[:]
        if not sky_wide:
            where_parts.insert(0, _cone_clause(ra, dec, radius, 'g.ra', 'g.dec'))
        where = ' AND '.join(where_parts) if where_parts else '1=1'
        adql  = (f'SELECT TOP {limit} {g_cols} '
                 f'FROM gaiadr3.gaia_source AS g '
                 f'JOIN gaiadr3.{join_table} AS x ON g.source_id = x.source_id '
                 f'WHERE {where} ORDER BY random_index')

    elif intent == 'nearest_neighbours':
        if 'parallax' not in columns:
            columns.insert(0, 'parallax')
        cols        = ', '.join(columns)
        where_parts = ['parallax > 0'] + filter_parts
        adql        = (f'SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source '
                       f'WHERE {" AND ".join(where_parts)} ORDER BY parallax DESC')

    elif intent == 'velocity_computation':
        forced = ['source_id', 'ra', 'dec', 'pmra', 'pmdec', 'radial_velocity']
        cols   = ', '.join(_resolve_columns(forced + columns))
        where_parts = ['radial_velocity IS NOT NULL'] + filter_parts
        if not sky_wide:
            where_parts.insert(0, _cone_clause(ra, dec, radius))
        adql = (f'SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source '
                f'WHERE {" AND ".join(where_parts)} ORDER BY random_index')

    else:
        raise ValueError(f"Unknown intent: '{intent}'")

    # Optional source_id injection for sequential pipeline steps
    if source_id_filter:
        ids  = ', '.join(str(int(i)) for i in source_id_filter[:200])
        adql = re.sub(r'\bWHERE\b',
                      f'WHERE source_id IN ({ids}) AND',
                      adql, count=1, flags=re.IGNORECASE)
        capped = len(source_id_filter) > 200
        print(f'  ↳ Injected source_id filter '
              f'({min(len(source_id_filter), 200)} IDs'
              f'{" capped" if capped else ""})')

    return adql


print('parser.py loaded.')
