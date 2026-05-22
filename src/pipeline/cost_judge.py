"""
cost_judge.py — LLM cost judge, fast pre-check, ADQL optimiser.
"""

from config import json, re
from model  import llm, JUDGE_SAMPLING_PARAMS

COST_JUDGE_SYSTEM_PROMPT = """You are an expert in ADQL queries on the Gaia DR3 database (1.8 billion rows).
Evaluate how expensive a query will be on the Gaia TAP server.

Return ONLY valid JSON, no markdown:
{
  "verdict":       "cheap" | "moderate" | "expensive" | "dangerous",
  "score":         0-100,
  "reasons":       [list of strings],
  "optimisations": [list of concrete ADQL rewrites],
  "safe_to_run":   true | false
}

Scoring guide:
   0-25  cheap     : TOP ≤ 1000, tight cone ≤ 0.5°, indexed filter, no JOIN
  26-50  moderate  : TOP ≤ 10000, cone ≤ 2°, simple filters
  51-75  expensive : TOP ≤ 50000, cone > 2°, ORDER BY non-indexed col, weak parallax
  76-100 dangerous : no TOP or TOP > 100000, no spatial filter, full-table scan, multi-JOIN

Key cost drivers:
  - Missing/large TOP (biggest risk)
  - Cone radius > 2 degrees
  - ORDER BY on parallax or radial_velocity (not indexed)
  - radial_velocity IS NOT NULL (only ~1% of stars, forces full scan)
  - parallax > 0  (matches ~90% of rows — very weak)
  - JOIN without a tight spatial filter
  - No WHERE clause

Optimisation strategies:
  - Reduce TOP N
  - Shrink cone radius
  - Replace parallax > 0 with parallax > 10 (nearby stars)
  - Add phot_g_mean_mag < 15 to cut faint stars early
  - Remove ORDER BY or replace with a parallax threshold filter
  - For sky-wide queries: suggest MOD(random_index, N) = 0 as a uniform sky sample.
    N=100 for moderate, N=1000 for expensive, N=10000 for dangerous."""


def _build_cost_prompt(adql: str) -> str:
    return (
        '<|im_start|>system\n'
        f'{COST_JUDGE_SYSTEM_PROMPT}<|im_end|>\n'
        '<|im_start|>user\n'
        f'Evaluate this ADQL query:\n\n{adql}<|im_end|>\n'
        '<|im_start|>assistant\n'
    )


def fast_cost_precheck(adql: str):
    """Rule-based fast check — no LLM call. Returns dict if dangerous, else None."""
    adql_upper = adql.upper()
    if 'CONTAINS' not in adql_upper and 'WHERE' not in adql_upper:
        return {
            'verdict': 'dangerous', 'score': 100, 'safe_to_run': False,
            'reasons': ['No WHERE clause — full scan of 1.8B rows.'],
            'optimisations': ['Add MOD(random_index, 10000) = 0 for a uniform sky sample.'],
        }
    top_match = re.search(r'\bTOP\s+(\d+)\b', adql, re.IGNORECASE)
    if not top_match:
        return {
            'verdict': 'dangerous', 'score': 95, 'safe_to_run': False,
            'reasons': ['No TOP clause — unbounded result set.'],
            'optimisations': ['Add TOP 1000 after SELECT.'],
        }
    if int(top_match.group(1)) > 100_000:
        return {
            'verdict': 'dangerous', 'score': 90, 'safe_to_run': False,
            'reasons': [f'TOP {int(top_match.group(1)):,} exceeds 100,000 row limit.'],
            'optimisations': ['Reduce TOP to ≤ 10,000.'],
        }
    return None


def evaluate_query_cost(adql: str) -> dict:
    """LLM judge: score and explain query cost."""
    prompt  = _build_cost_prompt(adql)
    outputs = llm.generate([prompt], JUDGE_SAMPLING_PARAMS)
    raw     = outputs[0].outputs[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'^```\s*',     '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$',     '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f'Cost judge returned non-JSON:\n{raw}')
        return {
            'verdict': 'expensive', 'score': 70, 'safe_to_run': False,
            'reasons': ['Cost judge failed to parse — defaulting conservative.'],
            'optimisations': [],
        }


def auto_optimise_adql(adql: str, optimisations: list,
                       top_cap: int = 5_000, score: int = 0) -> str:
    """Apply deterministic optimisations to the ADQL string."""
    has_cone     = 'CONTAINS' in adql.upper()
    has_sampling = 'random_index' in adql.lower()

    if not has_cone and not has_sampling:
        if score >= 75:    sample_n = 10000
        elif score >= 50:  sample_n = 1000
        else:              sample_n = 100
        adql = re.sub(r'\bWHERE\b',
                      f'WHERE MOD(random_index, {sample_n}) = 0 AND',
                      adql, count=1, flags=re.IGNORECASE)
        print(f'  Injected MOD(random_index, {sample_n}) — uniform sky sample')

    def cap_top(m):
        n = int(m.group(1))
        if n > top_cap:
            print(f'  TOP {n:,} → {top_cap:,}')
            return f'TOP {top_cap}'
        return m.group(0)
    adql = re.sub(r'\bTOP\s+(\d+)\b', cap_top, adql, flags=re.IGNORECASE)

    def shrink_cone(m):
        if float(m.group(3)) > 3.0:
            print(f'  Cone {m.group(3)}° → 2.0°')
            return f"CIRCLE('ICRS', {m.group(1)}, {m.group(2)}, 2.0)"
        return m.group(0)
    adql = re.sub(
        r"CIRCLE\s*\(\s*'ICRS'\s*,\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*,\s*([0-9.]+)\s*\)",
        shrink_cone, adql, flags=re.IGNORECASE
    )

    before = adql
    adql   = re.sub(r'\bparallax\s*>\s*0\b', 'parallax > 1', adql, flags=re.IGNORECASE)
    if adql != before:
        print('  parallax > 0  →  parallax > 1')

    if optimisations:
        print('  LLM judge also suggested:')
        for opt in optimisations:
            print(f'    • {opt}')
    return adql


_VERDICT_EMOJI = {'cheap': '🟢', 'moderate': '🟡', 'expensive': '🟠', 'dangerous': '🔴'}


def print_cost_report(cost: dict):
    verdict = cost.get('verdict', '?')
    score   = cost.get('score', '?')
    emoji   = _VERDICT_EMOJI.get(verdict, '⚪')
    print(f"  {'─'*52}")
    print(f"  {emoji}  COST: {verdict.upper()}  (score {score}/100)")
    print(f"  {'─'*52}")
    for r in cost.get('reasons', []):
        print(f'    ⚠️  {r}')
    for o in cost.get('optimisations', []):
        print(f'    💡 {o}')
    print(f"  {'─'*52}")


print('cost_judge.py loaded.')
