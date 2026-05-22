"""
complex_pipeline.py — Decomposer, plan validator, merged ADQL builder,
                       orchestrator, and complex_pipeline top-level entry.
"""

import json as _json_mod
import os   as _os_mod
import time as _time_mod

from config         import json, re, time, pd, MAX_RETRIES, OUTPUT_DIR
from model          import llm, DECOMPOSER_SAMPLING_PARAMS
from parser         import (VALID_COLUMNS, VALID_JOIN_TABLES, VALID_INTENTS,
                            _resolve_columns, _cone_clause, _filter_clauses,
                            _null_guards)
from cost_judge     import (fast_cost_precheck, evaluate_query_cost,
                            auto_optimise_adql, print_cost_report)
from tap            import run_query
from simple_pipeline import supervised_pipeline

# ── Decomposer ────────────────────────────────────────────────────────────────

DECOMPOSER_SYSTEM_PROMPT = f"""You are a query decomposer for an astronomy pipeline that queries the Gaia DR3 database.

A complex user query has already been identified as requiring multiple analyses.
Your job is to choose the right composition type and produce the minimal plan.

Return ONLY valid JSON — no markdown, no explanation.

COMPOSITION TYPES — choose exactly one:

  "merged"     — TWO OR MORE analyses on the SAME region that can be answered
                 by ONE ADQL query. This is the most efficient option.
                 Use when all intents are from this mergeable set:
                   cone_search, color_histogram, hr_diagram, stellar_population,
                   variability_search, velocity_computation
                 A merged plan has NO 'steps'. It has 'merged_intents' instead.
                 ALWAYS prefer merged over parallel when the region is the same.

  "parallel"   — N INDEPENDENT queries that CANNOT be merged because:
                   • they cover DIFFERENT sky regions, OR
                   • they use internal_crossmatch or nearest_neighbours
                 A parallel plan has a 'steps' list with no depends_on.

  "sequential" — N DEPENDENT queries where step N needs source_ids from step N-1.
                 A sequential plan has a 'steps' list where some steps have depends_on.

OUTPUT SCHEMA by composition:

Merged:
{{
  "shared_region": {{"target": string or null, "ra": float, "dec": float, "radius": float}},
  "composition":   "merged",
  "merged_intents": [list of intent strings],
  "columns":       [additional columns beyond each intent's defaults],
  "filters":       {{dict}},
  "limit":         int
}}

Parallel / Sequential:
{{
  "shared_region": {{"target": string or null, "ra": float, "dec": float, "radius": float}} or null,
  "composition":   "parallel" | "sequential",
  "steps": [
    {{
      "step_id":                int,
      "intent":                 string,
      "description":            string,
      "natural_language_query": string,
      "depends_on":             [list of step_id ints],
      "dependency_type":        "source_id_filter" or null,
      "override_region":        {{"ra": float, "dec": float, "radius": float}} or null,
      "filters":                {{dict}},
      "columns":                [list of strings],
      "join_table":             string or null,
      "limit":                  int
    }}
  ]
}}

FIELD RULES:
  shared_region   — required for merged; hoist here when steps share the same region.
  merged_intents  — present ONLY when composition = "merged". List of 2+ intent names.
  steps           — present ONLY when composition = "parallel" or "sequential".
  composition     — prefer "merged" over "parallel" when region is the same.
  depends_on      — [] for parallel. Sequential downstream steps list upstream step_id(s).
  dependency_type — null when depends_on empty; "source_id_filter" otherwise.
  override_region — set only when a step needs a DIFFERENT region from shared_region.
  limit           — merged default 2000. Steps with downstream dependants: ≤ 200.

Supported intents: {', '.join(sorted(VALID_INTENTS))}
Valid join_table values: {', '.join(sorted(VALID_JOIN_TABLES))}
Valid column names: {', '.join(sorted(VALID_COLUMNS))}

Rules:
  - PREFER merged when two or more analyses share the same region.
  - Decompose into the MINIMUM number of steps/intents.
  - Unknown targets: use galactic centre ra=266.4, dec=-29.0.
  - Steps with downstream dependants MUST include source_id in their columns.

Examples:

Query: "Show me the HR diagram and the colour histogram of the Pleiades"
{{"shared_region": {{"target": "Pleiades", "ra": 56.75, "dec": 24.12, "radius": 1.0}},
  "composition": "merged", "merged_intents": ["hr_diagram", "color_histogram"],
  "columns": [], "filters": {{}}, "limit": 2000}}

Query: "Compare colour distributions near Andromeda and the LMC"
{{"shared_region": null, "composition": "parallel", "steps": [
  {{"step_id": 1, "intent": "color_histogram", "description": "Andromeda",
    "natural_language_query": "Colour histogram of stars near Andromeda",
    "depends_on": [], "dependency_type": null,
    "override_region": {{"ra": 10.68, "dec": 41.27, "radius": 0.5}},
    "filters": {{}}, "columns": ["source_id","bp_rp","phot_g_mean_mag"], "join_table": null, "limit": 2000}},
  {{"step_id": 2, "intent": "color_histogram", "description": "LMC",
    "natural_language_query": "Colour histogram of stars near the LMC",
    "depends_on": [], "dependency_type": null,
    "override_region": {{"ra": 80.9, "dec": -69.8, "radius": 2.0}},
    "filters": {{}}, "columns": ["source_id","bp_rp","phot_g_mean_mag"], "join_table": null, "limit": 2000}}
]}}

Query: "Find variable stars near Orion, cross-match with Cepheid table, plot HR diagram"
{{"shared_region": {{"target": "Orion", "ra": 83.82, "dec": -5.39, "radius": 2.0}},
  "composition": "sequential", "steps": [
  {{"step_id": 1, "intent": "variability_search", "description": "Variable stars near Orion",
    "natural_language_query": "Variable stars near Orion",
    "depends_on": [], "dependency_type": null, "override_region": null,
    "filters": {{"phot_variable_flag": "VARIABLE"}},
    "columns": ["source_id","ra","dec","phot_g_mean_mag","phot_variable_flag"],
    "join_table": null, "limit": 200}},
  {{"step_id": 2, "intent": "internal_crossmatch", "description": "Crossmatch with Cepheids",
    "natural_language_query": "Cross-match stars near Orion with the Cepheid table",
    "depends_on": [1], "dependency_type": "source_id_filter", "override_region": null,
    "filters": {{}}, "columns": ["source_id","ra","dec","phot_g_mean_mag","bp_rp"],
    "join_table": "vari_cepheid", "limit": 200}},
  {{"step_id": 3, "intent": "hr_diagram", "description": "HR diagram of Cepheids",
    "natural_language_query": "HR diagram of Cepheid variable stars near Orion",
    "depends_on": [2], "dependency_type": "source_id_filter", "override_region": null,
    "filters": {{}}, "columns": ["source_id","parallax","phot_g_mean_mag","bp_rp"],
    "join_table": null, "limit": 1000}}
]}}
"""


def _build_decomposer_prompt(user_query: str) -> str:
    return (
        '<|im_start|>system\n'
        f'{DECOMPOSER_SYSTEM_PROMPT}<|im_end|>\n'
        '<|im_start|>user\n'
        f'{user_query}<|im_end|>\n'
        '<|im_start|>assistant\n'
    )


def _build_decomposer_correction_prompt(original_query, bad_plan, errors, attempt):
    error_list = '\n'.join(f'  - {e}' for e in errors)
    return (
        f'Your previous plan for "{original_query}" had errors '
        f'(attempt {attempt}/{MAX_RETRIES}):\n'
        f'{error_list}\n\n'
        f'Bad plan:\n{json.dumps(bad_plan, indent=2)}\n\n'
        f'Fix all errors and return ONLY valid JSON matching the required schema.'
    )


def decompose_query(user_query: str) -> dict:
    """Call the LLM to decompose a complex query into a plan. Raises RuntimeError on failure."""
    current_prompt = user_query
    last_plan = {}

    for attempt in range(1, MAX_RETRIES + 1):
        print(f'  [Decomposer attempt {attempt}/{MAX_RETRIES}] …')
        prompt  = _build_decomposer_prompt(current_prompt)
        outputs = llm.generate([prompt], DECOMPOSER_SAMPLING_PARAMS)
        raw     = outputs[0].outputs[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'^```\s*',     '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\s*```$',     '', raw)

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            print(f'    Non-JSON from decomposer:\n{raw[:200]}')
            current_prompt = (
                f'Your answer for "{user_query}" was not valid JSON.\n'
                f'Return ONLY a valid JSON object matching the required schema.'
            )
            last_plan = {}
            continue

        last_plan = plan
        errors    = validate_plan(plan)

        if not errors:
            composition = plan['composition']
            if composition == 'merged':
                n = len(plan.get('merged_intents', []))
                print(f'    Plan valid. {n} merged intents, composition=merged')
            else:
                n = len(plan.get('steps', []))
                print(f'    Plan valid. {n} steps, composition={composition}')
            return plan

        print(f'    Plan errors ({len(errors)}):')
        for e in errors:
            print(f'      • {e}')
        current_prompt = _build_decomposer_correction_prompt(
            user_query, plan, errors, attempt
        )

    raise RuntimeError(
        f'Decomposer failed after {MAX_RETRIES} attempts for: "{user_query}"\n'
        f'Last plan:\n{json.dumps(last_plan, indent=2)}'
    )


# ── Plan validator ────────────────────────────────────────────────────────────

_PLAN_CONE_REQUIRED = {'cone_search', 'color_histogram', 'variability_search'}


def validate_plan(plan: dict) -> list:
    """Return list of error strings. Empty list means the plan is valid."""
    errors = []
    if not isinstance(plan, dict):
        return ['Plan must be a JSON object.']

    composition = plan.get('composition')
    if composition not in ('parallel', 'sequential', 'merged'):
        errors.append(
            f"'composition' must be 'parallel', 'sequential', or 'merged', "
            f"got: {composition!r}"
        )
        return errors

    if composition == 'merged':
        merged_intents = plan.get('merged_intents')
        if not isinstance(merged_intents, list) or len(merged_intents) < 2:
            errors.append("'merged_intents' must be a list of at least 2 intent strings.")
        shared_region = plan.get('shared_region')
        if not shared_region:
            errors.append("merged plan requires 'shared_region' with ra/dec/radius.")
        else:
            for field in ('ra', 'dec', 'radius'):
                if shared_region.get(field) is None:
                    errors.append(f"shared_region.{field} must not be null.")
        if plan.get('steps') is not None:
            errors.append("merged plan must not have a 'steps' field.")
        return errors

    steps = plan.get('steps')
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("'steps' must be a non-empty list.")
        return errors

    shared_region = plan.get('shared_region')
    if shared_region is not None:
        for field in ('ra', 'dec', 'radius'):
            if shared_region.get(field) is None:
                errors.append(f"shared_region.{field} must not be null.")

    seen_ids = set()
    for step in steps:
        sid = step.get('step_id')
        if not isinstance(sid, int):
            errors.append(f"step_id must be an integer, got: {sid!r}")
        elif sid in seen_ids:
            errors.append(f"Duplicate step_id: {sid}")
        else:
            seen_ids.add(sid)

        intent = step.get('intent')
        if intent not in VALID_INTENTS:
            errors.append(f"Step {sid}: intent {intent!r} is not valid.")

        nlq = step.get('natural_language_query', '')
        if not isinstance(nlq, str) or not nlq.strip():
            errors.append(f"Step {sid}: 'natural_language_query' must be a non-empty string.")

        depends_on      = step.get('depends_on', [])
        dependency_type = step.get('dependency_type')
        if not isinstance(depends_on, list):
            errors.append(f"Step {sid}: 'depends_on' must be a list.")
            depends_on = []
        if depends_on and dependency_type != 'source_id_filter':
            errors.append(f"Step {sid}: 'dependency_type' must be 'source_id_filter' when depends_on is non-empty.")
        if not depends_on and dependency_type is not None:
            errors.append(f"Step {sid}: 'dependency_type' must be null when depends_on is empty.")

        join_table = step.get('join_table')
        if intent == 'internal_crossmatch':
            if not join_table:
                errors.append(f"Step {sid}: 'join_table' required for internal_crossmatch.")
            elif join_table not in VALID_JOIN_TABLES:
                errors.append(f"Step {sid}: join_table {join_table!r} is not a valid DR3 table.")
        elif join_table is not None:
            errors.append(f"Step {sid}: 'join_table' must be null for intent {intent!r}.")

        if intent in _PLAN_CONE_REQUIRED:
            if shared_region is None and step.get('override_region') is None:
                errors.append(f"Step {sid}: intent {intent!r} requires shared_region or override_region.")

        override = step.get('override_region')
        if override is not None:
            for field in ('ra', 'dec', 'radius'):
                if override.get(field) is None:
                    errors.append(f"Step {sid}: override_region.{field} must not be null.")

        limit = step.get('limit', 1000)
        if not isinstance(limit, int) or limit <= 0:
            errors.append(f"Step {sid}: 'limit' must be a positive integer.")

    for step in steps:
        for dep_id in step.get('depends_on', []):
            if dep_id not in seen_ids:
                errors.append(f"Step {step['step_id']}: depends_on references unknown step_id {dep_id}.")

    # Cycle detection (Kahn's)
    adj       = {s['step_id']: set(s.get('depends_on', [])) for s in steps}
    in_degree = {sid: 0 for sid in seen_ids}
    for sid, deps in adj.items():
        for d in deps:
            if d in in_degree:
                in_degree[sid] += 1
    queue   = [sid for sid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for sid in seen_ids:
            if node in adj.get(sid, set()):
                in_degree[sid] -= 1
                if in_degree[sid] == 0:
                    queue.append(sid)
    if visited != len(seen_ids):
        errors.append('Cycle detected in step dependencies.')

    all_empty = all(len(s.get('depends_on', [])) == 0 for s in steps)
    any_set   = any(len(s.get('depends_on', [])) > 0 for s in steps)
    if composition == 'parallel' and not all_empty:
        errors.append("composition='parallel' but some steps have depends_on.")
    if composition == 'sequential' and not any_set:
        errors.append("composition='sequential' but no steps have depends_on.")

    steps_with_dependants = {dep_id for s in steps for dep_id in s.get('depends_on', [])}
    step_map = {s['step_id']: s for s in steps}
    for sid in steps_with_dependants:
        if sid in step_map and 'source_id' not in step_map[sid].get('columns', []):
            errors.append(f"Step {sid} has downstream dependants but 'source_id' is not in its columns.")

    return errors


# ── Merged ADQL builder ───────────────────────────────────────────────────────

_INTENT_META = {
    'cone_search':        {'forced_cols': ['source_id', 'ra', 'dec'],
                           'where_guards': [], 'select_extra': [],
                           'incompatible': {'internal_crossmatch', 'nearest_neighbours'}},
    'color_histogram':    {'forced_cols': ['source_id', 'bp_rp', 'phot_g_mean_mag',
                                           'phot_bp_mean_mag', 'phot_rp_mean_mag'],
                           'where_guards': ['bp_rp IS NOT NULL'], 'select_extra': [],
                           'incompatible': {'internal_crossmatch', 'nearest_neighbours'}},
    'hr_diagram':         {'forced_cols': ['source_id', 'parallax', 'phot_g_mean_mag', 'bp_rp'],
                           'where_guards': ['parallax > 0', 'bp_rp IS NOT NULL'],
                           'select_extra': ['(phot_g_mean_mag + 5*LOG10(parallax/100)) AS abs_g_mag'],
                           'incompatible': {'internal_crossmatch', 'nearest_neighbours'}},
    'stellar_population': {'forced_cols': ['source_id', 'ra', 'dec'],
                           'where_guards': [], 'select_extra': [],
                           'incompatible': {'internal_crossmatch', 'nearest_neighbours'}},
    'variability_search': {'forced_cols': ['source_id', 'ra', 'dec', 'phot_variable_flag'],
                           'where_guards': ["phot_variable_flag = 'VARIABLE'"], 'select_extra': [],
                           'incompatible': {'internal_crossmatch', 'nearest_neighbours'}},
    'velocity_computation': {'forced_cols': ['source_id', 'ra', 'dec', 'pmra', 'pmdec', 'radial_velocity'],
                              'where_guards': ['radial_velocity IS NOT NULL'], 'select_extra': [],
                              'incompatible': {'internal_crossmatch', 'nearest_neighbours'}},
}
_MERGEABLE_INTENTS = set(_INTENT_META.keys())


def build_adql_merged(plan: dict) -> str:
    intents = plan.get('merged_intents', [])
    if not intents:
        raise ValueError("build_adql_merged: 'merged_intents' is empty.")
    for intent in intents:
        if intent not in _MERGEABLE_INTENTS:
            raise ValueError(f"build_adql_merged: intent '{intent}' is not mergeable.")
    intent_set = set(intents)
    for intent in intents:
        bad = _INTENT_META[intent]['incompatible'] & intent_set
        if bad:
            raise ValueError(f"build_adql_merged: '{intent}' incompatible with {bad}.")

    region = plan.get('shared_region') or {}
    ra     = region.get('ra')
    dec    = region.get('dec')
    radius = region.get('radius', 1.0)
    if ra is None or dec is None:
        raise ValueError("build_adql_merged: shared_region with ra/dec required.")

    limit   = plan.get('limit', 2000)
    filters = plan.get('filters') or {}

    base_cols = ['source_id', 'ra', 'dec', 'phot_g_mean_mag', 'bp_rp', 'pmra', 'pmdec']
    all_cols  = list(base_cols)
    for intent in intents:
        for col in _INTENT_META[intent]['forced_cols']:
            if col not in all_cols:
                all_cols.append(col)
    for col in _resolve_columns(plan.get('columns') or []):
        if col not in all_cols:
            all_cols.append(col)

    seen_aliases, select_extras = set(), []
    for intent in intents:
        for expr in _INTENT_META[intent]['select_extra']:
            alias = expr.split(' AS ')[-1].strip() if ' AS ' in expr else expr
            if alias not in seen_aliases:
                seen_aliases.add(alias)
                select_extras.append(expr)

    select_cols = ', '.join(all_cols)
    if select_extras:
        select_cols += ', ' + ', '.join(select_extras)

    where_parts = [_cone_clause(ra, dec, radius)]
    seen_guards = set()
    for intent in intents:
        for guard in _INTENT_META[intent]['where_guards']:
            if guard not in seen_guards:
                seen_guards.add(guard)
                where_parts.append(guard)
    for guard in _null_guards(all_cols):
        if guard not in where_parts:
            where_parts.append(guard)
    where_parts.extend(_filter_clauses(filters))

    return (
        f'SELECT TOP {limit} {select_cols} '
        f'FROM gaiadr3.gaia_source '
        f'WHERE {" AND ".join(where_parts)} '
        f'ORDER BY random_index'
    )


def validate_merged_plan(plan: dict) -> list:
    errors  = []
    intents = plan.get('merged_intents', [])
    if not intents:
        errors.append("merged plan must have a non-empty 'merged_intents' list.")
        return errors
    for intent in intents:
        if intent not in _MERGEABLE_INTENTS:
            errors.append(f"Intent '{intent}' cannot be merged.")
    intent_set = set(intents)
    for intent in intents:
        bad = _INTENT_META.get(intent, {}).get('incompatible', set()) & intent_set
        if bad:
            errors.append(f"Intent '{intent}' is incompatible with {bad}.")
    if not plan.get('shared_region'):
        errors.append("merged plan requires 'shared_region' with ra/dec/radius.")
    else:
        for field in ('ra', 'dec', 'radius'):
            if plan['shared_region'].get(field) is None:
                errors.append(f"shared_region.{field} must not be null.")
    return errors


# ── Orchestrator ──────────────────────────────────────────────────────────────

STEP_RETRIES = 2


def _effective_query(step: dict, plan: dict) -> str:
    nlq      = step['natural_language_query']
    override = step.get('override_region')
    if override:
        nlq = (f'{nlq} (ra={override["ra"]}, dec={override["dec"]}, '
               f'radius={override["radius"]} deg)')
    return nlq


def _build_step_correction_prompt(original_query, step, error, attempt):
    return (
        f'The following Gaia DR3 query failed (attempt {attempt}/{STEP_RETRIES}):\n'
        f'  Original: "{original_query}"\n'
        f'  Intent: {step["intent"]}\n'
        f'  Error: {error}\n\n'
        f'Rewrite to fix the error. Keep the same intent and region. '
        f'Return only the corrected natural-language query string.'
    )


def _execute_step(step: dict, plan: dict, source_id_filter: list = None,
                   skip_gaia_execution: bool = False) -> tuple:
    base_query    = _effective_query(step, plan)
    current_query = base_query
    last_error    = None
    t0            = time.time()

    for attempt in range(1, STEP_RETRIES + 1):
        if attempt > 1:
            print(f'    [Step retry {attempt}/{STEP_RETRIES}]')
            current_query = _build_step_correction_prompt(
                base_query, step, str(last_error), attempt
            )
        t0 = time.time()
        try:
            result   = supervised_pipeline(current_query, source_id_filter=source_id_filter)
            df       = result['results'][1]
            duration = round(time.time() - t0, 2)
            return df, {'status': 'success', 'rows': len(df), 'duration': duration}
        except RuntimeError as exc:
            last_error = exc
            duration   = round(time.time() - t0, 2)
            if 'DANGEROUS' in str(exc) or 'dangerous' in str(exc).lower():
                print(f'  🔴  Step {step["step_id"]} blocked by cost judge.')
                return None, {'status': 'blocked', 'error': str(exc), 'duration': duration}
            print(f'  ✗  Step {step["step_id"]} failed (attempt {attempt}): {exc}')
        except Exception as exc:
            last_error = exc
            print(f'  ✗  Step {step["step_id"]} unexpected error (attempt {attempt}): {exc}')

    return None, {'status': 'failed', 'error': str(last_error),
                  'duration': round(time.time() - t0, 2)}


def _topological_order(steps: list) -> list:
    id_to_step = {s['step_id']: s for s in steps}
    in_degree  = {s['step_id']: len(s.get('depends_on', [])) for s in steps}
    queue      = sorted([sid for sid, deg in in_degree.items() if deg == 0])
    order      = []
    while queue:
        sid = queue.pop(0)
        order.append(id_to_step[sid])
        for step in steps:
            if sid in step.get('depends_on', []):
                in_degree[step['step_id']] -= 1
                if in_degree[step['step_id']] == 0:
                    queue.append(step['step_id'])
                    queue.sort()
    return order


def run_merged(plan: dict) -> tuple:
    print(f'\n[Merged] Intents: {plan["merged_intents"]}')
    t0 = time.time()
    try:
        adql = build_adql_merged(plan)
    except ValueError as exc:
        return {'merged': None}, {'merged': {'status': 'failed', 'error': str(exc), 'duration': 0, 'adql': None}}, ['merged']

    print(f'  ADQL: {adql}')

    # Validate ADQL before cost judge (shared validation step)
    from validator import validate_adql
    adql_errors = validate_adql(adql)
    if adql_errors:
        print(f'  ADQL errors: {adql_errors}')
        return ({'merged': None},
                {'merged': {'status': 'failed', 'error': str(adql_errors), 'adql': adql, 'duration': round(time.time()-t0,2)}},
                ['merged'])

    print('  Evaluating cost …')
    cost = fast_cost_precheck(adql)
    if cost is None:
        cost = evaluate_query_cost(adql)
    print_cost_report(cost)

    if cost['verdict'] == 'dangerous':
        print('  🔴  Blocked by cost judge.')
        return ({'merged': None},
                {'merged': {'status': 'blocked', 'adql': adql,
                            'error': f"Dangerous (score {cost['score']})",
                            'duration': round(time.time()-t0, 2)}},
                ['merged'])

    if not cost['safe_to_run']:
        adql = auto_optimise_adql(adql, cost.get('optimisations', []), score=cost.get('score', 0))
        print(f'  Optimised ADQL: {adql}')

    try:
        df       = run_query(adql)
        duration = round(time.time() - t0, 2)
        print(f'  ✓  {len(df)} rows in {duration}s')
        return {'merged': df}, {'merged': {'status': 'success', 'adql': adql, 'rows': len(df), 'duration': duration}}, []
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        return {'merged': None}, {'merged': {'status': 'failed', 'adql': adql, 'error': str(exc), 'duration': duration}}, ['merged']


def run_parallel(plan: dict) -> tuple:
    results, timings, failed_ids = {}, {}, []
    for step in plan['steps']:
        sid = step['step_id']
        print(f'\n[Step {sid}] {step["description"]}')
        print(f'  Query: {_effective_query(step, plan)}')
        df, timing   = _execute_step(step, plan)
        results[sid] = df
        timings[sid] = timing
        if timing['status'] == 'success':
            print(f'  ✓  {timing["rows"]} rows in {timing["duration"]}s')
        else:
            failed_ids.append(sid)
    return results, timings, failed_ids


def run_sequential(plan: dict) -> tuple:
    results, timings, failed_ids = {}, {}, []
    for step in _topological_order(plan['steps']):
        sid        = step['step_id']
        depends_on = step.get('depends_on', [])
        print(f'\n[Step {sid}] {step["description"]}')
        print(f'  Query: {_effective_query(step, plan)}')

        skip_reason = None
        for dep_id in depends_on:
            if timings.get(dep_id, {}).get('status') != 'success':
                skip_reason = f'upstream step {dep_id} did not succeed'
                break
            if 'source_id' not in (results.get(dep_id) or pd.DataFrame()).columns:
                skip_reason = f'upstream step {dep_id} has no source_id column'
                break
        if skip_reason:
            print(f'  ⊘  Skipping: {skip_reason}')
            results[sid] = None
            timings[sid] = {'status': 'skipped', 'reason': skip_reason, 'duration': 0}
            failed_ids.append(sid)
            continue

        source_ids = None
        if depends_on:
            ids = []
            for dep_id in depends_on:
                ids.extend(results[dep_id]['source_id'].dropna().astype(int).tolist())
            source_ids = list(dict.fromkeys(ids))
            print(f'  ↳  {len(source_ids)} source_ids from step(s) {depends_on}')

        df, timing   = _execute_step(step, plan, source_id_filter=source_ids)
        results[sid] = df
        timings[sid] = timing
        if timing['status'] == 'success':
            print(f'  ✓  {timing["rows"]} rows in {timing["duration"]}s')
        else:
            failed_ids.append(sid)

    return results, timings, failed_ids


def orchestrate(plan: dict) -> tuple:
    composition = plan['composition']
    n = len(plan.get('steps') or plan.get('merged_intents', []))
    print(f"\n{'─'*60}")
    print(f'  ORCHESTRATOR  {composition.upper()}  {n} intent(s)/step(s)')
    print(f"{'─'*60}")
    if composition == 'merged':   return run_merged(plan)
    elif composition == 'parallel': return run_parallel(plan)
    else:                           return run_sequential(plan)


# ── Complex pipeline ──────────────────────────────────────────────────────────

PLAN_RETRIES = 2


def _build_plan_correction_prompt(user_query, plan, failed_ids, timings, attempt):
    lines = [
        f'The plan for "{user_query}" had {len(failed_ids)} failing step(s) '
        f'(outer attempt {attempt}/{PLAN_RETRIES}).', '', 'Failed steps/keys:',
    ]
    for sid in failed_ids:
        t = timings.get(sid, {})
        lines.append(f'  {sid} — {t.get("status","unknown")}: {t.get("error") or t.get("reason","unknown")}')
    lines += ['', 'Previous plan:', json.dumps(plan, indent=2), '',
              'Revise to fix failing steps. Keep successful ones unchanged. '
              'Return ONLY valid JSON matching the required schema.']
    return '\n'.join(lines)


def build_output_json(user_query, plan, timings, outer_attempt):
    composition = plan['composition']
    if composition == 'merged':
        t = timings.get('merged', {})
        execution  = [{'key': 'merged', 'intents': plan.get('merged_intents', []),
                        'status': t.get('status', 'unknown'), 'adql': t.get('adql'),
                        'rows_returned': t.get('rows', 0), 'duration_seconds': t.get('duration', 0),
                        'error': t.get('error')}]
        successful = 1 if t.get('status') == 'success' else 0
        failed     = 0 if successful else 1
        skipped    = 0
    else:
        execution = []
        for step in plan['steps']:
            sid = step['step_id']
            t   = timings.get(sid, {})
            execution.append({'step_id': sid, 'intent': step['intent'],
                               'description': step['description'],
                               'status': t.get('status', 'unknown'), 'adql': t.get('adql'),
                               'rows_returned': t.get('rows', 0),
                               'duration_seconds': t.get('duration', 0),
                               'error': t.get('error') or t.get('reason')})
        successful = sum(1 for e in execution if e['status'] == 'success')
        failed     = sum(1 for e in execution if e['status'] in ('failed', 'blocked'))
        skipped    = sum(1 for e in execution if e['status'] == 'skipped')

    total_dur = round(sum(e['duration_seconds'] for e in execution), 2)
    return {
        'query': user_query, 'routed_as': 'complex', 'composition': composition,
        'outer_attempt': outer_attempt, 'plan': plan, 'execution': execution,
        'summary': {'total_steps': len(execution), 'successful': successful,
                    'failed': failed, 'skipped': skipped,
                    'total_duration_seconds': total_dur},
    }


def complex_pipeline(user_query: str, save_json: bool = True, skip_gaia_execution: bool = False) -> dict:
    """
    Full complex pipeline with outer plan-level retry loop.
    Returns {'output_json': ..., 'results': ...}.
    """
    print(f"\n{'='*60}")
    print(f'  COMPLEX PIPELINE\n  Query: {user_query}')
    print(f"{'='*60}\n")

    decompose_prompt = user_query
    last_output, last_results = None, {}

    for outer_attempt in range(1, PLAN_RETRIES + 1):
        if outer_attempt > 1:
            print(f'\n[Outer retry {outer_attempt}/{PLAN_RETRIES}] Re-planning …')

        print(f'[{outer_attempt}.1] Decomposing …')
        try:
            plan = decompose_query(decompose_prompt)
        except RuntimeError as exc:
            raise RuntimeError(f'Decomposer failed: {exc}') from exc

        if plan.get('composition') == 'merged':
            merge_errors = validate_merged_plan(plan)
            if merge_errors:
                raise RuntimeError(f'Merged plan invalid: {merge_errors}')

        composition = plan['composition']
        if composition == 'merged':
            print(f'\n  Plan (merged): {plan.get("merged_intents")} '
                  f'over {plan.get("shared_region", {}).get("target", "region")}\n')
        else:
            print(f'\n  Plan ({composition}, {len(plan["steps"])} step(s)):')
            for s in plan['steps']:
                dep = f' → depends on {s["depends_on"]}' if s.get('depends_on') else ''
                print(f'    Step {s["step_id"]}: {s["intent"]:25} {s["description"][:40]}{dep}')
            print()

        print(f'[{outer_attempt}.2] Executing …')
        results, timings, failed_ids = orchestrate(plan)

        output       = build_output_json(user_query, plan, timings, outer_attempt)
        last_output  = output
        last_results = results
        s = output['summary']
        print(f'\n  Attempt {outer_attempt}: ✓ {s["successful"]}  ✗ {s["failed"]}  '
              f'⊘ {s["skipped"]}  ({s["total_duration_seconds"]}s)')

        if not failed_ids:
            print('  All steps succeeded.')
            break
        if outer_attempt < PLAN_RETRIES:
            print(f'  {failed_ids} failed — revising plan …')
            decompose_prompt = _build_plan_correction_prompt(
                user_query, plan, failed_ids, timings, outer_attempt
            )
        else:
            print(f'  {failed_ids} still failing — returning partial results.')

    if save_json and last_output:
        safe_name = re.sub(r'[^a-z0-9]+', '_', user_query.lower())[:60]
        out_path  = _os_mod.path.join(OUTPUT_DIR, f'{safe_name}.json')
        with open(out_path, 'w') as fh:
            _json_mod.dump(last_output, fh, indent=2)
        print(f'\n  Output JSON → {out_path}')

    s = last_output['summary']
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({last_output['composition'].upper()}, attempt {last_output['outer_attempt']})")
    print(f"  Steps: {s['total_steps']}  ✓ {s['successful']}  ✗ {s['failed']}  ⊘ {s['skipped']}")
    print(f"  Time : {s['total_duration_seconds']}s")
    print(f"{'='*60}\n")

    return {'output_json': last_output, 'results': last_results}


print('complex_pipeline.py loaded.')
