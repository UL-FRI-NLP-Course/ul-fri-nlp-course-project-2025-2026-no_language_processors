"""Cost checking utilities for ADQL queries."""
import re

def fast_cost_precheck(adql: str):
    adql_upper = adql.upper()
    if 'CONTAINS' not in adql_upper and 'WHERE' not in adql_upper:
        return {
            'verdict': 'dangerous', 'score': 100, 'safe_to_run': False,
            'reasons': ['No WHERE clause — full scan.'],
            'optimisations': ['Add MOD(random_index, 10000) = 0']
        }
    top_match = re.search(r'\bTOP\s+(\d+)\b', adql, re.IGNORECASE)
    if not top_match:
        return {'verdict':'dangerous','score':95,'safe_to_run':False,'reasons':['No TOP clause'],'optimisations':['Add TOP 1000']}
    if int(top_match.group(1)) > 100_000:
        return {'verdict':'dangerous','score':90,'safe_to_run':False,'reasons':[f'TOP {int(top_match.group(1))} too large'],'optimisations':['Reduce TOP']}
    return None
