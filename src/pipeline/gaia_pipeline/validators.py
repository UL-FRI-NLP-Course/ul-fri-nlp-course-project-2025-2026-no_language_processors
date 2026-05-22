"""Validation utilities for parsed JSON and ADQL."""
import re

VALID_INTENTS = {
    'cone_search','color_histogram','hr_diagram','stellar_population',
    'variability_search','internal_crossmatch','nearest_neighbours','velocity_computation',
}

VALID_COLUMNS = {
    'ra', 'dec', 'source_id', 'parallax', 'parallax_error',
    'pmra', 'pmdec', 'radial_velocity',
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp',
    'teff_gspphot', 'logg_gspphot', 'mh_gspphot',
    'phot_variable_flag', 'ruwe',
}

def validate_parsed_json(q: dict) -> list:
    errors = []
    intent = q.get('intent')
    if not intent or intent not in VALID_INTENTS:
        errors.append('invalid intent')
        return errors
    # spatial checks simplified
    if intent in {'cone_search','color_histogram','variability_search'}:
        for f in ('ra','dec'):
            if q.get(f) is None:
                errors.append(f'{f} required for intent {intent}')
    limit = q.get('limit',1000)
    if not isinstance(limit, int) or limit<=0:
        errors.append('limit must be positive int')
    cols = q.get('columns') or []
    unknown = [c for c in cols if c not in VALID_COLUMNS]
    if unknown:
        errors.append(f'unknown columns: {unknown}')
    return errors

def validate_adql(adql: str) -> list:
    errors = []
    if re.search(r'\bNone\b', adql):
        errors.append("ADQL contains 'None'")
    if not re.search(r'\bTOP\s+\d+\b', adql, re.IGNORECASE):
        errors.append('No TOP clause')
    if not re.search(r'\bWHERE\b', adql, re.IGNORECASE):
        errors.append('No WHERE clause')
    return errors
