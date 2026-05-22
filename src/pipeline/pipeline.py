"""
pipeline.py — Top-level entry point.

Usage in notebook:
    from pipeline import routed_pipeline
    result = routed_pipeline("Show me the HR diagram of the Pleiades")
"""

from config          import display
from router          import route_query
from simple_pipeline import supervised_pipeline
from complex_pipeline import complex_pipeline


def routed_pipeline(user_query: str, skip_gaia_execution: bool = False) -> dict:
    """
    Route then dispatch:
      simple  → supervised_pipeline
      complex → complex_pipeline

    Args:
      user_query (str): natural language question
      skip_gaia_execution (bool): if True, skip Gaia TAP calls and return mock data

    Returns {'output_json': ..., 'results': ...} from whichever pipeline ran.
    """
    print(f"\n{'═'*60}")
    print(f'  USER QUERY: {user_query}')
    print(f"{'═'*60}")

    print('\n[Router] Classifying query …')
    routing = route_query(user_query)
    verdict = routing['complexity']
    reason  = routing['reason']

    icon = '🟢' if verdict == 'simple' else '🟣'
    print(f'  {icon}  Verdict: {verdict.upper()}')
    print(f'      Reason: {reason}\n')

    if verdict == 'simple':
        return supervised_pipeline(user_query, skip_gaia_execution=skip_gaia_execution)
    else:
        return complex_pipeline(user_query, skip_gaia_execution=skip_gaia_execution)


print('pipeline.py loaded. Use routed_pipeline(query) to run.')
