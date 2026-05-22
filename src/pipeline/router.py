"""
router.py — LLM router: classifies a query as simple or complex.
"""

from config import json, re
from model  import llm, SAMPLING_PARAMS

ROUTER_SYSTEM_PROMPT = """
You are a router for an astronomy query pipeline. Your job is to decide whether a user query can be answered with a SINGLE Gaia database query, or whether it requires MULTIPLE queries.

Return ONLY valid JSON, no markdown, no explanation:
{
  "complexity": "simple" | "complex",
  "reason": "<one short sentence>"
}

A query is SIMPLE if it requests one analysis, one region, one population.
A query is COMPLEX only if it explicitly asks for multiple distinct outputs, multiple regions to compare, or sequential operations where one result feeds another.
PREFER "simple" WHEN IN DOUBT.

Examples:
Query: "Show me stars around the Pleiades"
{"complexity": "simple", "reason": "single cone search"}

Query: "HR diagram of nearby red dwarfs"
{"complexity": "simple", "reason": "one HR diagram with a population filter"}

Query: "Show me the HR diagram AND the colour histogram of the Pleiades"
{"complexity": "complex", "reason": "two distinct analyses of one region"}

Query: "Compare colour distributions near Andromeda and the LMC"
{"complexity": "complex", "reason": "same analysis in two different regions"}

Query: "Find variable stars near Orion and compute their kinematics"
{"complexity": "complex", "reason": "two analyses chained on the same population"}

Query: "Show me red dwarfs and also white dwarfs"
{"complexity": "complex", "reason": "two distinct populations requested"}

Query: "What's the temperature distribution of metal-poor giants near M31"
{"complexity": "simple", "reason": "one distribution of one filtered population"}
"""


def _build_router_prompt(user_query: str) -> str:
    return (
        '<|im_start|>system\n'
        f'{ROUTER_SYSTEM_PROMPT}<|im_end|>\n'
        '<|im_start|>user\n'
        f'{user_query}<|im_end|>\n'
        '<|im_start|>assistant\n'
    )


def route_query(user_query: str) -> dict:
    """Classify a query as simple or complex. Returns {'complexity': ..., 'reason': ...}."""
    prompt  = _build_router_prompt(user_query)
    outputs = llm.generate([prompt], SAMPLING_PARAMS)
    raw     = outputs[0].outputs[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'^```\s*',     '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$',     '', raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f'  Router returned non-JSON, defaulting to simple:\n{raw}')
        return {'complexity': 'simple', 'reason': 'router fallback'}
    if result.get('complexity') not in ('simple', 'complex'):
        print('  Router returned invalid complexity, defaulting to simple')
        return {'complexity': 'simple', 'reason': 'router fallback'}
    return result


print('router.py loaded.')
