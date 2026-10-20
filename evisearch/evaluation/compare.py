"""Paired comparison of current arms on dev2/test (CPU): python -m evisearch.evaluation.compare TARGET BASELINE [BASELINE ...]
Writes runs/release/compare-<target>.json and prints a table: EM, F1, full-evidence rate, EM | full evidence,
early stopping (incomplete evidence with searches < 3), mean searches, and paired bootstrap of EM / early stop
for the target minus each baseline (5,000 resamples; question uncertainty only, not seed uncertainty).
"""
import json
import random
import statistics
import sys

from evisearch.artifacts import RUNS
from resources import read_jsonl
from evisearch.evaluation.behavior import predictions

SPLITS = ('dev2', 'test')


def stats(m, ids):
    full = [i for i in ids if m[i]['evidence_coverage'] >= 1]
    early = lambda x: float(x['evidence_coverage'] < 1 and x['search_calls'] < 3 and x['format_valid'])
    return dict(em=statistics.fmean(m[i]['em'] for i in ids), f1=statistics.fmean(m[i]['f1'] for i in ids),
                full_rate=len(full) / len(ids), em_given_full=statistics.fmean(m[i]['em'] for i in full) if full else None,
                early_stop=statistics.fmean(early(m[i]) for i in ids), search=statistics.fmean(m[i]['search_calls'] for i in ids),
                format=statistics.fmean(float(m[i]['format_valid']) for i in ids))


def paired(a, b, ids, f):
    d = [f(a[i]) - f(b[i]) for i in ids]
    random.seed(0)
    s = sorted(statistics.fmean(random.choices(d, k=len(d))) for _ in range(5000))
    return [statistics.fmean(d), s[125], s[4875]]


def main(target, *baselines):
    assert baselines and len(set((target, *baselines))) == 1 + len(baselines), 'Choose distinct target and baseline runs'
    result = {}
    for split in SPLITS:
        models = {m: {r['id']: r['metrics'] for r in predictions(m, split)} for m in (target, *baselines)}
        configs = [json.loads((RUNS / 'eval' / m / split / 'config.json').read_text()) for m in models]
        for key in ('dataset_sha256', 'decoding', 'max_trajectory', 'max_turn_tokens', 'source_sha256', 'protocol_sha256'):
            assert len({c[key] for c in configs}) == 1, f'Comparison protocol differs: {key}'
        ids = sorted(models[target])
        assert all(set(v) == set(ids) for v in models.values()), split
        early = lambda x: float(x['evidence_coverage'] < 1 and x['search_calls'] < 3 and x['format_valid'])
        result[split] = dict(stats={m: stats(v, ids) for m, v in models.items()},
                             paired={b: dict(em=paired(models[target], models[b], ids, lambda x: x['em']),
                                             early_stop=paired(models[target], models[b], ids, early))
                                     for b in baselines})
    (RUNS / f'compare-{target}.json').write_text(json.dumps(result, indent=1) + '\n')
    for split, v in result.items():
        print(f'== {split}')
        for key in ('em', 'f1', 'full_rate', 'em_given_full', 'early_stop', 'search', 'format'):
            values = []
            for m, s in v['stats'].items():
                value = s[key]
                shown = 'n/a' if value is None else f'{value * 100 if key != "search" else value:.2f}'
                values.append(f'{m} {shown}')
            print(f'  {key:13s}', '  '.join(values))
        for b, p in v['paired'].items():
            print(f'  {target} - {b}: EM {p["em"][0]*100:+.2f} [{p["em"][1]*100:+.2f}, {p["em"][2]*100:+.2f}]'
                  f'  early stop {p["early_stop"][0]*100:+.2f} [{p["early_stop"][1]*100:+.2f}, {p["early_stop"][2]*100:+.2f}]')


if __name__ == '__main__':
    main(*sys.argv[1:])
