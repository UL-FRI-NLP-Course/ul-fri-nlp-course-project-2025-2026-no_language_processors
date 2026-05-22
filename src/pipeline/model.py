"""
model.py — Load the vLLM model exactly once per session.

The `if 'llm' not in globals()` guard ensures the GPU model is not
reloaded when this module is re-imported or a notebook cell is re-run.

All other modules import `llm`, `SAMPLING_PARAMS`, and
`JUDGE_SAMPLING_PARAMS` from here.
"""

from vllm import LLM, SamplingParams
from config import HF_CACHE, os

MODEL_ID             = 'Qwen/Qwen2.5-7B-Instruct'
TENSOR_PARALLEL_SIZE = int(os.environ.get('TENSOR_PARALLEL_SIZE', '1'))
DTYPE                = 'float16'

# ── Load once, reuse across re-imports ───────────────────────────────────────
if 'llm' not in globals():
    print(f'Loading {MODEL_ID} from {HF_CACHE} …')
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype=DTYPE,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    print('Model loaded.')
else:
    print('Model already loaded — reusing.')

SAMPLING_PARAMS       = SamplingParams(temperature=0.0, max_tokens=512)
JUDGE_SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=512)
DECOMPOSER_SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=1024)
