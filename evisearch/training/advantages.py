"""IGPO equations 3–7; isolated arithmetic for deterministic CPU verification.

Reference: GuoqingWang1/IGPO, commit 64165e2741ed8801f977948c8128080ce87b4101.
Upstream core_algos.py uses population variance + 1e-8, followed by epsilon 1e-6.
"""
import math
import statistics
import string


def normalize_group(values):
    if not values:
        return []
    mean = statistics.mean(values)
    std = math.sqrt(statistics.pvariance(values) + 1e-8) if len(values) > 1 else 1.0
    return [(v - mean) / (std + 1e-6) for v in values]


def answer_f1(prediction, aliases):
    """Match official IGPO token-set F1, including punctuation-to-space normalization."""
    table = str.maketrans({c: ' ' for c in string.punctuation})
    pred = set(prediction.lower().translate(table).split())
    return max((2 * len(pred & gold) / (len(pred) + len(gold)) if pred and gold else 0.0
                for gold in (set(a.lower().translate(table).split()) for a in aliases)), default=0.0)


def assistant_spans(mask):
    """Half-open ranges of sampled assistant tokens separated by tool suffixes."""
    spans, start = [], None
    for i, value in enumerate(list(mask) + [0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    return spans


def turn_advantages(outcomes, logps, mode, gamma=1.0):
    """logps contains before-search and subsequent observation-prefix scores per rollout."""
    if mode not in ('prob_joint', 'prob_separate', 'igpo'):
        raise ValueError(mode)
    signals = []
    for row in logps:
        values = row if mode == 'igpo' else [math.exp(v) for v in row]
        signals.append([b - a for a, b in zip(values, values[1:])])
    flat = [v for row in signals for v in row]
    if mode == 'prob_joint':
        joint = normalize_group(flat + list(outcomes))
        norm_ig, norm_outcome = joint[:len(flat)], joint[len(flat):]
    else:
        norm_ig, norm_outcome = normalize_group(flat), normalize_group(outcomes)
    offset, returns, normalized = 0, [], []
    for raw, end in zip(signals, norm_outcome, strict=True):
        row = norm_ig[offset:offset + len(raw)] + [end]
        offset += len(raw)
        normalized.append(row)
        result, value = [], 0.0
        for reward in reversed(row):
            value = reward + gamma * value
            result.append(value)
        returns.append(list(reversed(result)))
    return returns, signals, normalized


def overlong_penalty(generated_tokens, truncated, soft_start=768, hard_limit=1024):
    """DAPO linear soft penalty on policy-generated tokens; observations excluded."""
    if truncated:
        return -1.0
    return -min(1.0, max(0.0, (generated_tokens - soft_start) / (hard_limit - soft_start)))
