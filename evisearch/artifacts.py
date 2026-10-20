"""Frozen artifact identities, separated from the current source layout.

The numbered directories below identify existing immutable experiment artifacts,
not Python packages. No historical checkout is imported or needed at runtime.
New training, generated data and evaluation always use separate destinations.
"""
import hashlib
import re
from pathlib import Path

from resources import ROOT

CODE = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/v3'
PROTOCOL = ROOT / 'data/v4'
SEEDS = ROOT / 'data/v2'
TEACHER_SFT = DATA / 'sft.jsonl'
FROZEN_SFT = ROOT / 'data/v5/sft_err.jsonl'
PREPARED = ROOT / 'data/release-prepared'
RUNS = ROOT / 'runs/release'

# Explicit provenance for the measurements in the public report. These are
# recorded runs, not runs produced by the extracted publication code.
RECORDED = {
    'base': ROOT / 'runs/v3/eval/base',
    'teacher-sft': ROOT / 'runs/v3/eval/sft-42',
    'rationale-sft': ROOT / 'runs/v5/eval/sft_err',
    'grpo': ROOT / 'runs/v5/eval/grpo_err-42',
    'igpo': ROOT / 'runs/v5/eval/igpo_err-42',
    'vcr': ROOT / 'runs/v5/eval/vcr-42',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_name(value):
    """A run identifier must not change the output root."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('Run name must start with a letter/digit and contain only letters, digits, dot, dash or underscore')
    return value


def source_files():
    return sorted((CODE / 'evisearch').rglob('*.py')) + [
        CODE / 'resources.py', CODE / 'environment.freeze.txt',
        *sorted((CODE / 'configs').glob('*.json')),
    ]
