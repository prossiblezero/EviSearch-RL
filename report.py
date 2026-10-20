"""Recompute current recorded comparisons from actual predictions (CPU only).

This reads recorded experiment artifacts, never runs a model, and never claims
the publication refactor has been retrained. Missing tasks fail closed.
"""
import json

from resources import read_jsonl
from evisearch.artifacts import CODE, DATA, RECORDED, sha
from evisearch.evaluation.compare import paired, stats
from evisearch.evaluation.rollout import aggregate


def main():
    result = {'provenance': 'Recorded experiments before source extraction; single training seed.', 'splits': {}}
    for split in ('dev2', 'test'):
        tasks = read_jsonl(DATA / f'{split}_tasks.jsonl')
        expected = {t['id'] for t in tasks if t['domain'] == 'general'}
        assert len(expected) == 512
        models, hashes, configs = {}, {}, []
        for model in ('grpo', 'igpo', 'vcr'):
            path = RECORDED[model] / split / 'predictions.jsonl'
            rows = [r for r in read_jsonl(path) if r['mode'] == 'agent' and r['domain'] == 'general']
            assert len(rows) == len(expected) and {r['id'] for r in rows} == expected, (model, split)
            config = json.loads((path.parent / 'config.json').read_text())
            summary = json.loads((path.parent / 'summary.json').read_text())
            assert summary['config'] == config and summary['predictions_sha256'] == sha(path)
            assert config['dataset_sha256'] == sha(DATA / f'{split}_tasks.jsonl')
            assert config['decoding'] == 'greedy' and config['max_trajectory'] == 6144 and config['max_turn_tokens'] == 512
            models[model] = {r['id']: r['metrics'] for r in rows}
            hashes[model] = sha(path)
            configs.append(config)
        for key in ('dataset_sha256', 'source_sha256', 'wrapper_sha256', 'protocol_sha256', 'decoding', 'max_trajectory', 'max_turn_tokens'):
            assert len({c[key] for c in configs}) == 1, key
        ids = sorted(expected)
        early = lambda x: float(x['evidence_coverage'] < 1 and x['search_calls'] < 3 and x['format_valid'])
        result['splits'][split] = dict(
            tasks=len(ids), dataset_sha256=sha(DATA / f'{split}_tasks.jsonl'), predictions_sha256=hashes,
            stats={m: stats(v, ids) for m, v in models.items()},
            vcr_minus_grpo={
                'em': paired(models['vcr'], models['grpo'], ids, lambda x: x['em']),
                'early_stop': paired(models['vcr'], models['grpo'], ids, early),
            })
    # Base/SFT must be compared within the same retrieval mode. These baseline
    # runs are part of the current evidence, not reports for previous environments.
    baselines, static_prompts = {}, {}
    for model in ('base', 'teacher-sft'):
        path = RECORDED[model] / 'test' / 'predictions.jsonl'
        rows = read_jsonl(path)
        summary = json.loads((path.parent / 'summary.json').read_text())
        assert summary['predictions_sha256'] == sha(path)
        config = json.loads((path.parent / 'config.json').read_text())
        assert summary['config'] == config
        assert config['dataset_sha256'] == sha(DATA / 'test_tasks.jsonl')
        assert config['decoding'] == 'greedy' and config['max_trajectory'] == 6144 and config['max_turn_tokens'] == 512
        for mode in ('agent', 'static'):
            selected = [r for r in rows if r['mode'] == mode and r['domain'] == 'general']
            assert len(selected) == len(expected) and {r['id'] for r in selected} == expected
        static_prompts[model] = {r['id']: r['messages'][:2] for r in rows if r['mode'] == 'static'}
        baselines[model] = dict(predictions_sha256=sha(path), results=aggregate(rows))
    assert static_prompts['base'] == static_prompts['teacher-sft'], 'Static evidence differs between Base and SFT'
    result['test_baselines'] = baselines
    result['limitations'] = [
        'The test split was reused and inspected during iterative research; it is not a fresh blind test.',
        'Bootstrap intervals cover task sampling, not training-seed uncertainty.',
        'The rationale-SFT checkpoint has dev2 evaluation but no complete recorded test evaluation.',
    ]
    path = CODE / 'results/current.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('Recorded test results (not a retraining of this checkout):')
    for name, value in result['splits']['test']['stats'].items():
        print(f'{name:5s} EM={100*value["em"]:.2f}% F1={100*value["f1"]:.2f}% early_stop={100*value["early_stop"]:.2f}%')
    print('Wrote', path)


if __name__ == '__main__':
    main()
