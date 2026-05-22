"""Utility helpers for gaia pipeline."""
import re
import os
import json
from typing import List

def safe_name(text: str, maxlen: int = 60) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', text.lower())[:maxlen]
    return s

def dump_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2)

def resolve_column_alias(col: str, aliases: dict) -> str:
    return aliases.get(col, col)
