"""
eval_metrics.py — Multiple scoring metrics for ADQL query evaluation.

Metrics:
  - bleu4_score:         n-gram overlap (0-1)
  - exact_match:         1.0 if queries are identical, 0.0 otherwise
  - token_f1:            precision & recall of SQL tokens
  - structural_match:    similarity of query structure
  - complexity_match:    does generated complexity match expected
"""

import re
import math
from collections import Counter


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_adql(query: str) -> list:
    """
    SQL-aware tokenizer for ADQL queries.
    Lowercases and splits on whitespace + SQL punctuation.
    """
    query = query.lower().strip()
    tokens = re.findall(r"[a-z0-9_.]+|[=<>!(),;']", query)
    return tokens


def extract_sql_elements(query: str) -> dict:
    """Extract structured elements from an ADQL query."""
    query_upper = query.upper()
    
    # Extract SELECT columns
    select_match = re.search(r'SELECT\s+(?:TOP\s+\d+\s+)?(.+?)\s+FROM', query_upper)
    select_cols = set()
    if select_match:
        cols_str = select_match.group(1)
        cols = re.findall(r'\b[a-z_]+\b', cols_str.lower())
        select_cols = set(cols)
    
    # Extract FROM table
    from_match = re.search(r'FROM\s+([a-z0-9_.]+)', query_upper)
    from_table = from_match.group(1).lower() if from_match else None
    
    # Extract WHERE conditions
    where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', query_upper)
    where_conditions = []
    if where_match:
        where_clause = where_match.group(1)
        # Split by AND
        conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        where_conditions = [c.strip() for c in conditions if c.strip()]
    
    # Extract JOIN info
    join_match = re.search(r'JOIN\s+([a-z0-9_.]+)', query_upper)
    join_table = join_match.group(1).lower() if join_match else None
    
    # Detect CONTAINS (cone search)
    has_cone = 'CONTAINS' in query_upper
    
    # Extract TOP N
    top_match = re.search(r'TOP\s+(\d+)', query_upper)
    top_n = int(top_match.group(1)) if top_match else None
    
    return {
        'select_cols': select_cols,
        'from_table': from_table,
        'where_conditions': where_conditions,
        'join_table': join_table,
        'has_cone': has_cone,
        'top_n': top_n,
    }


# ── BLEU Score (from earlier, kept for compatibility) ────────────────────────

def _ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """BLEU-N score (0-1)."""
    if not hypothesis or not hypothesis.strip():
        return 0.0
    
    hyp_tokens = tokenize_adql(hypothesis)
    ref_tokens = tokenize_adql(reference)
    
    if not hyp_tokens:
        return 0.0
    
    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_ngrams(hyp_tokens, n))
        ref_ng = Counter(_ngrams(ref_tokens, n))
        if not hyp_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(count, ref_ng[gram]) for gram, count in hyp_ng.items())
        total = sum(hyp_ng.values())
        precisions.append((clipped + 1) / (total + 1))  # add-1 smoothing
    
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_avg), 4)


# ── Exact Match ───────────────────────────────────────────────────────────────

def exact_match(hypothesis: str, reference: str) -> float:
    """
    Exact match score: 1.0 if queries are identical (after normalization), 0.0 otherwise.
    Normalize by removing extra whitespace and lowercasing.
    """
    hyp_norm = ' '.join(hypothesis.lower().split())
    ref_norm = ' '.join(reference.lower().split())
    return 1.0 if hyp_norm == ref_norm else 0.0


# ── Token F1 ──────────────────────────────────────────────────────────────────

def token_f1(hypothesis: str, reference: str) -> float:
    """
    Compute F1 score based on token overlap.
    Precision: what fraction of hyp tokens appear in ref
    Recall: what fraction of ref tokens appear in hyp
    """
    hyp_tokens = set(tokenize_adql(hypothesis))
    ref_tokens = set(tokenize_adql(reference))
    
    if not hyp_tokens and not ref_tokens:
        return 1.0  # both empty
    if not hyp_tokens or not ref_tokens:
        return 0.0  # one empty, one not
    
    intersection = hyp_tokens & ref_tokens
    precision = len(intersection) / len(hyp_tokens) if hyp_tokens else 0.0
    recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


# ── Structural Match ──────────────────────────────────────────────────────────

def structural_match(hypothesis: str, reference: str) -> float:
    """
    Score based on matching query structure:
    - Same FROM table (high weight: 0.3)
    - Same SELECT columns (high weight: 0.3)
    - Same number of WHERE conditions (medium weight: 0.2)
    - Same JOIN table (if present, weight: 0.1)
    - Both have/don't have CONTAINS (cone search, weight: 0.1)
    """
    hyp_struct = extract_sql_elements(hypothesis)
    ref_struct = extract_sql_elements(reference)
    
    score = 0.0
    
    # FROM table (0.3)
    if hyp_struct['from_table'] == ref_struct['from_table']:
        score += 0.3
    
    # SELECT columns overlap (0.3)
    hyp_cols = hyp_struct['select_cols']
    ref_cols = ref_struct['select_cols']
    if hyp_cols and ref_cols:
        col_overlap = len(hyp_cols & ref_cols) / max(len(hyp_cols | ref_cols), 1)
        score += 0.3 * col_overlap
    
    # WHERE conditions count (0.2)
    hyp_where_count = len(hyp_struct['where_conditions'])
    ref_where_count = len(hyp_struct['where_conditions'])
    if hyp_where_count > 0 and ref_where_count > 0:
        # Partial credit if counts are close
        count_diff = abs(hyp_where_count - ref_where_count)
        condition_score = max(0, 1.0 - (count_diff / max(hyp_where_count, ref_where_count)))
        score += 0.2 * condition_score
    
    # JOIN table (0.1)
    if hyp_struct['join_table'] and ref_struct['join_table']:
        if hyp_struct['join_table'] == ref_struct['join_table']:
            score += 0.1
    elif not hyp_struct['join_table'] and not ref_struct['join_table']:
        score += 0.1
    
    # CONTAINS (cone search) (0.1)
    if hyp_struct['has_cone'] == ref_struct['has_cone']:
        score += 0.1
    
    return round(score, 4)


# ── Complexity Match ──────────────────────────────────────────────────────────

def complexity_match(generated_query: str, gold_query: str, 
                     expected_complexity: str = None) -> dict:
    """
    Estimate complexity based on query features:
    - Simple: cone_search, single condition
    - Medium: multiple conditions, filters
    - Complex: JOINs, aggregations
    
    Returns {'estimated': 'simple'|'medium'|'complex', 'match': 0.0-1.0}
    """
    def estimate_complexity(query: str) -> str:
        query_upper = query.upper()
        has_join = 'JOIN' in query_upper
        has_group = 'GROUP BY' in query_upper
        has_agg = any(f in query_upper for f in ['COUNT(', 'AVG(', 'MAX(', 'MIN(', 'SUM('])
        
        if has_join or has_group or has_agg:
            return 'complex'
        elif 'AND' in query_upper:
            return 'medium'
        else:
            return 'simple'
    
    gen_complexity = estimate_complexity(generated_query)
    gold_complexity = estimate_complexity(gold_query)
    
    match_score = 1.0 if gen_complexity == gold_complexity else 0.5
    
    return {
        'generated': gen_complexity,
        'gold': gold_complexity,
        'match': match_score,
    }


# ── Combined Score ────────────────────────────────────────────────────────────

def combined_score(hypothesis: str, reference: str, 
                   weights: dict = None) -> dict:
    """
    Compute weighted combination of all metrics.
    
    Default weights:
    - bleu4: 0.35 (overall n-gram quality)
    - token_f1: 0.25 (token-level precision/recall)
    - structural: 0.25 (query structure)
    - exact: 0.15 (perfect match bonus)
    """
    if weights is None:
        weights = {
            'bleu4': 0.35,
            'token_f1': 0.25,
            'structural': 0.25,
            'exact': 0.15,
        }
    
    scores = {
        'bleu4': bleu_score(hypothesis, reference),
        'token_f1': token_f1(hypothesis, reference),
        'structural': structural_match(hypothesis, reference),
        'exact': exact_match(hypothesis, reference),
    }
    
    combined = sum(scores[k] * weights[k] for k in weights.keys())
    
    return {
        'scores': scores,
        'combined': round(combined, 4),
        'weights': weights,
    }


print('eval_metrics.py loaded.')
