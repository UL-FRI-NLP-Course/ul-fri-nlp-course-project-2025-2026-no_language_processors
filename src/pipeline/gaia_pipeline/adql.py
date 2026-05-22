"""ADQL builder extracted from the notebook."""
import re
from .utils import resolve_column_alias

_COLUMN_ALIASES = {
    'magnitude': 'phot_g_mean_mag',
    'g_mag': 'phot_g_mean_mag',
}

VALID_COLUMNS = {
    'ra', 'dec', 'source_id', 'parallax', 'parallax_error',
    'pmra', 'pmdec', 'radial_velocity',
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp',
    'teff_gspphot', 'logg_gspphot', 'mh_gspphot',
    'phot_variable_flag', 'ruwe',
}

def _resolve_columns(cols):
    if not cols:
        return ['source_id','ra','dec']
    resolved = [ _COLUMN_ALIASES.get(c, c) for c in cols ]
    out = []
    seen = set()
    for c in resolved:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _cone_clause(ra, dec, radius, ra_col='ra', dec_col='dec'):
    return f"1=CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), CIRCLE('ICRS', {ra}, {dec}, {radius}))"

def _filter_clauses(filters: dict):
    clauses = []
    for col, val in (filters or {}).items():
        col = _COLUMN_ALIASES.get(col, col)
        if isinstance(val, dict):
            mn, mx = val.get('min'), val.get('max')
            if mn is not None and mx is not None:
                clauses.append(f"{col} BETWEEN {mn} AND {mx}")
            elif mn is not None:
                clauses.append(f"{col} >= {mn}")
            elif mx is not None:
                clauses.append(f"{col} <= {mx}")
        elif isinstance(val, str):
            clauses.append(f"{col} = '{val}'")
        else:
            clauses.append(f"{col} = {val}")
    return clauses

def build_adql(parsed: dict) -> str:
    intent = parsed.get('intent')
    ra = parsed.get('ra')
    dec = parsed.get('dec')
    radius = parsed.get('radius', 1.0)
    limit = parsed.get('limit', 1000)
    filters = parsed.get('filters') or {}
    columns = _resolve_columns(parsed.get('columns') or [])

    # ensure basic columns
    for c in ('source_id','ra','dec'):
        if c not in columns:
            columns.insert(0,c)

    cols = ', '.join(columns)
    filter_parts = _filter_clauses(filters)

    if intent == 'cone_search':
        where = ' AND '.join([_cone_clause(ra, dec, radius)] + filter_parts)
        return f"SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {where} ORDER BY random_index"

    if intent == 'nearest_neighbours':
        where = ' AND '.join(filter_parts) or '1=1'
        return f"SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {where} ORDER BY parallax DESC"

    # fallback: generic select with filters
    where = ' AND '.join(filter_parts) if filter_parts else '1=1'
    return f"SELECT TOP {limit} {cols} FROM gaiadr3.gaia_source WHERE {where} ORDER BY random_index"
