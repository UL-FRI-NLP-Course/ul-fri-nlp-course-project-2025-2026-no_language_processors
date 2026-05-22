"""
simple_pipeline.py — supervised_pipeline (simple path).

Parse → validate JSON → build ADQL → validate ADQL → cost judge → run_query.
Returns the same {'output_json': ..., 'results': ...} dict as complex_pipeline.
"""

import json as _json_mod
import os   as _os_mod
import time as _time_mod

from config      import json, re, MAX_RETRIES, OUTPUT_DIR, pd
from parser      import parse_query, build_adql
from validator   import validate_parsed_json, validate_adql, build_correction_prompt
from cost_judge  import fast_cost_precheck, evaluate_query_cost, auto_optimise_adql, print_cost_report
from tap         import run_query


def supervised_pipeline(user_query: str, source_id_filter: list = None,
                        skip_gaia_execution: bool = False) -> dict:
    """
    Full simple pipeline with retry loop.

    Args:
      user_query (str): natural language question
      source_id_filter (list, optional): source IDs to inject as WHERE IN filter
      skip_gaia_execution (bool): if True, skip TAP query and return mock DataFrame

    Returns:
      {
        'output_json': { query, routed_as, composition, plan, execution, summary },
        'results':     { 1: DataFrame (empty if skip_gaia_execution=True) }
      }
    Raises RuntimeError after MAX_RETRIES exhausted.
    """
    print(f"\n{'='*60}")
    print(f'  USER QUERY: {user_query}')
    if source_id_filter:
        print(f'  SOURCE_ID FILTER: {len(source_id_filter)} IDs supplied')
    print(f"{'='*60}\n")

    current_prompt = user_query
    parsed         = None
    adql           = None
    t0             = _time_mod.time()

    for attempt in range(1, MAX_RETRIES + 1):

        # 1 — LLM → JSON
        print(f'[Attempt {attempt}/{MAX_RETRIES}] Parsing …')
        try:
            parsed = parse_query(current_prompt)
        except ValueError as exc:
            print(f'  Non-JSON from LLM: {exc}')
            current_prompt = (
                f'Your answer was not valid JSON.\n'
                f'Original query: "{user_query}"\n'
                f'Error: {exc}\n'
                f'Return ONLY a valid JSON object.'
            )
            continue
        print(f'  Parsed JSON:\n{json.dumps(parsed, indent=2)}')

        # 2 — validate JSON
        json_errors = validate_parsed_json(parsed)
        if json_errors:
            print(f'  JSON errors ({len(json_errors)}):')
            for e in json_errors:
                print(f'    • {e}')
            current_prompt = build_correction_prompt(
                user_query, parsed, json_errors, attempt
            )
            continue
        print('  JSON valid.')

        # 3 — build ADQL (source_id injection handled inside build_adql)
        try:
            adql = build_adql(parsed, source_id_filter=source_id_filter)
        except (ValueError, KeyError) as exc:
            print(f'  build_adql() failed: {exc}')
            current_prompt = build_correction_prompt(
                user_query, parsed, [f'build_adql() error: {exc}'], attempt
            )
            continue
        print(f'  ADQL: {adql}')

        # 4 — validate ADQL
        adql_errors = validate_adql(adql)
        if adql_errors:
            print(f'  ADQL errors ({len(adql_errors)}):')
            for e in adql_errors:
                print(f'    • {e}')
            current_prompt = build_correction_prompt(
                user_query, parsed, adql_errors, attempt
            )
            continue
        print('  ADQL syntax valid.')

        # 5 — cost gate
        print('  Evaluating cost …')
        cost = fast_cost_precheck(adql)
        if cost is None:
            cost = evaluate_query_cost(adql)
        print_cost_report(cost)

        if cost['verdict'] == 'dangerous':
            print('  🔴 BLOCKED — query too dangerous.')
            current_prompt = build_correction_prompt(
                user_query, parsed,
                [f"Query rated DANGEROUS (score {cost['score']}/100). "
                 f"Reasons: {'; '.join(cost['reasons'])}. "
                 f"Fix: {'; '.join(cost['optimisations'])}"],
                attempt,
            )
            continue

        if not cost['safe_to_run']:
            adql = auto_optimise_adql(
                adql, cost.get('optimisations', []), score=cost.get('score', 0)
            )
            print(f'  Optimised ADQL: {adql}')

        # 6 — execute (or mock if skipping Gaia)
        try:
            if skip_gaia_execution:
                print(f'  ⊘  Gaia execution skipped (flag: skip_gaia_execution=True)')
                df       = pd.DataFrame()  # empty DataFrame
                duration = round(_time_mod.time() - t0, 2)
            else:
                df       = run_query(adql)
                duration = round(_time_mod.time() - t0, 2)

            output = {
                'query':         user_query,
                'routed_as':     'simple',
                'composition':   'simple',
                'outer_attempt': 1,
                'plan': {
                    'intent':     parsed.get('intent'),
                    'ra':         parsed.get('ra'),
                    'dec':        parsed.get('dec'),
                    'radius':     parsed.get('radius'),
                    'filters':    parsed.get('filters', {}),
                    'join_table': parsed.get('join_table'),
                    'limit':      parsed.get('limit'),
                    'source_id_filter_size': len(source_id_filter) if source_id_filter else 0,
                },
                'execution': [{
                    'step_id':          1,
                    'intent':           parsed.get('intent'),
                    'status':           'success',
                    'adql':             adql,
                    'rows_returned':    len(df),
                    'duration_seconds': duration,
                    'error':            None,
                }],
                'summary': {
                    'total_steps':            1,
                    'successful':             1,
                    'failed':                 0,
                    'skipped':                0,
                    'total_duration_seconds': duration,
                },
            }

            safe_name = re.sub(r'[^a-z0-9]+', '_', user_query.lower())[:60]
            out_path  = _os_mod.path.join(OUTPUT_DIR, f'{safe_name}.json')
            with open(out_path, 'w') as fh:
                _json_mod.dump(output, fh, indent=2)

            print(f'\n  ✓  {len(df)} rows in {duration}s')
            print(f'  Output JSON → {out_path}')
            return {'output_json': output, 'results': {1: df}}

        except Exception as exc:
            print(f'  run_query() failed: {exc}')
            if '500' in str(exc):
                current_prompt = build_correction_prompt(
                    user_query, parsed,
                    [f'ADQL caused HTTP 500 on Gaia TAP: {exc}'],
                    attempt,
                )
                continue
            raise

    raise RuntimeError(
        f'Pipeline failed after {MAX_RETRIES} attempts for query: "{user_query}"'
    )


print('simple_pipeline.py loaded.')
