"""
config.py — Environment setup, SSL fix, standard imports.
Import this first in every other module.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
os.environ.setdefault('CURL_CA_BUNDLE', '')
os.environ.setdefault('REQUESTS_CA_BUNDLE', '')

import json
import re
import time
import numpy as np
import pandas as pd
from IPython.display import display

# ── HuggingFace cache ─────────────────────────────────────────────────────────
HF_CACHE = os.path.abspath('../hf_cache')
os.environ['HF_HOME']            = HF_CACHE
os.environ['HF_HUB_CACHE']      = os.path.join(HF_CACHE, 'hub')
os.environ['HF_HUB_DISABLE_XET'] = '1'

# ── Pipeline output directory ─────────────────────────────────────────────────
OUTPUT_DIR = './pipeline_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared retry constant ─────────────────────────────────────────────────────
MAX_RETRIES = 3

print(f'config.py loaded. HF cache: {HF_CACHE}')
