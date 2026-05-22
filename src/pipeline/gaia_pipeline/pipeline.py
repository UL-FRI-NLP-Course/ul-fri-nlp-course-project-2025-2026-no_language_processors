"""High-level pipeline functions adapted from the notebook.

This module provides supervised_pipeline, complex_pipeline (stubbed to call orchestrator),
and a routed_pipeline entrypoint suitable for notebooks.
"""
import time
from . import adql, gaia_client, validators, cost, utils

MAX_RETRIES = 3

def parse_query_stub(user_query: str) -> dict:
    """Placeholder parser. For production, wire to LLM client.

    For now we implement a couple of heuristics for demo/testing.
    """
    q = {}
    # simple heuristics
    if 'pleiades' in user_query.lower():
        q['intent'] = 'cone_search'
        q['ra'] = 56.75
        q['dec'] = 24.12
        q['radius'] = 1.0
    elif 'nearest' in user_query.lower():
        q['intent'] = 'nearest_neighbours'
        q['ra'] = None
        q['dec'] = None
    else:
        q['intent'] = 'cone_search'
        q['ra'] = 266.4
        q['dec'] = -29.0
        q['radius'] = 0.5
    q['columns'] = ['source_id','ra','dec','phot_g_mean_mag','bp_rp']
    q['filters'] = {}
    q['limit'] = 100
    return q


def supervised_pipeline(user_query: str, source_id_filter: list = None):
    print('\n'+'='*60)
    print(f'  USER QUERY: {user_query}')
    print('='*60+'\n')

    parsed = parse_query_stub(user_query)
    errors = validators.validate_parsed_json(parsed)
    if errors:
        raise RuntimeError(f'Parsed JSON invalid: {errors}')

    adql_str = adql.build_adql(parsed)
    adql_errors = validators.validate_adql(adql_str)
    if adql_errors:
        raise RuntimeError(f'ADQL invalid: {adql_errors}')

    c = cost.fast_cost_precheck(adql_str)
    if c and c.get('verdict') == 'dangerous':
        raise RuntimeError('Query judged dangerous')

    # run
    df = gaia_client.run_query(adql_str)
    return df


def routed_pipeline(user_query: str):
    # naive router: everything simple for now
    return supervised_pipeline(user_query)


def complex_pipeline(user_query: str):
    # For now fallback to supervised
    return supervised_pipeline(user_query)
