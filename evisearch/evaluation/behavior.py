"""Audit distractor recovery on saved agent predictions."""
import json
import statistics
import sys

from evisearch.artifacts import DATA, RUNS
from resources import ROOT, read_jsonl, setup_paths
from evisearch.environment import EvidenceEnv


def predictions(model, split):
    from evisearch.artifacts import RUNS, sha, run_name
    if split not in ('dev', 'dev2', 'test', 'medical'):
        raise ValueError('Unknown frozen split')
    path = RUNS / 'eval' / run_name(model) / split / 'predictions.jsonl'
    config = json.loads((path.parent / 'config.json').read_text())
    summary = json.loads((path.parent / 'summary.json').read_text())
    assert summary['config'] == config
    assert summary['predictions_sha256'] == sha(path), 'Predictions changed after evaluation'
    tasks_path = DATA / f'{split}_tasks.jsonl'
    expected = {t['id'] for t in read_jsonl(tasks_path)}
    assert config['dataset_sha256'] == sha(tasks_path)
    assert config['decoding'] == 'greedy' and config['max_trajectory'] == 6144 and config['max_turn_tokens'] == 512
    rows = [r for r in read_jsonl(path) if r['mode'] == 'agent']
    assert rows and len(rows) == len(expected) and {r['id'] for r in rows} == expected, 'Incomplete or duplicate predictions'
    return rows


def analyse(rows, tasks):
    env, out = EvidenceEnv(), []
    for r in rows:
        task = tasks[r['id']]
        if task['domain'] != 'general':
            continue
        env.reset(task=task)
        source = {alias: d['id'] for alias, d in env._by_id.items()}
        gold = set(task['support_ids'])
        calls = r['calls']
        reads = [i for i, c in enumerate(calls) if c['name'] == 'read' and c.get('ok')
                 and isinstance(c.get('arguments'), dict) and source.get(c['arguments'].get('doc_id')) not in gold]
        m = r['metrics']
        row = dict(full=m['evidence_coverage'] >= 1, em=m['em'], distractor=bool(reads))
        if reads:
            alias = calls[reads[0]]['arguments']['doc_id']
            after = calls[reads[0] + 1:]
            row.update(saved=any(c['name'] == 'add_evidence' and isinstance(c.get('arguments'), dict)
                                 and str(c['arguments'].get('passage_id', '')).split(':')[0] == alias for c in after),
                       read_again=any(c['name'] == 'read' for c in after),
                       searched_again=any(c['name'] == 'search' for c in after))
        out.append(row)
    d = [x for x in out if x['distractor']]
    mean = lambda v: statistics.fmean(v) if v else None
    return dict(n=len(out), distractor_read_rate=len(d) / len(out),
                distractor_saved_rate=mean([x['saved'] for x in d]),
                recovered_full_after_distractor=mean([x['full'] for x in d]),
                em_after_distractor=mean([x['em'] for x in d]),
                read_again_after_distractor=mean([x['read_again'] for x in d]),
                searched_again_after_distractor=mean([x['searched_again'] for x in d]),
                full_rate=mean([x['full'] for x in out]), em=mean([x['em'] for x in out]))


def main(split, *models):
    setup_paths()
    tasks = {t['id']: t for t in read_jsonl(DATA / f'{split}_tasks.jsonl')}
    result = {m: analyse(predictions(m, split), tasks) for m in models}
    (RUNS / f'recovery-stats-{split}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
