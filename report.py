"""Recompute current recorded comparisons from actual predictions (CPU only).

This reads recorded experiment artifacts, never runs a model, and never claims
the publication refactor has been retrained. Missing tasks fail closed.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from resources import read_jsonl
from evisearch.artifacts import CODE, DATA, RECORDED, RECORDED_9B, TRAINED_2B, TRAINED_9B, sha
from evisearch.evaluation.compare import paired, stats
from evisearch.evaluation.rollout import aggregate

PROTOCOL_KEYS = ('dataset_sha256', 'source_sha256', 'wrapper_sha256', 'protocol_sha256', 'decoding',
                 'max_trajectory', 'max_turn_tokens')
early = lambda x: float(x['evidence_coverage'] < 1 and x['search_calls'] < 3 and x['format_valid'])
full = lambda x: float(x['evidence_coverage'] >= 1)
em = lambda x: x['em']


def load_agent(root, split, expected):
    """Agent/general metrics of one recorded evaluation, with its config and prediction hash."""
    path = root / split / 'predictions.jsonl'
    rows = [r for r in read_jsonl(path) if r['mode'] == 'agent' and r['domain'] == 'general']
    assert len(rows) == len(expected) and {r['id'] for r in rows} == expected, (root, split)
    config = json.loads((path.parent / 'config.json').read_text())
    summary = json.loads((path.parent / 'summary.json').read_text())
    assert summary['config'] == config and summary['predictions_sha256'] == sha(path)
    assert config['dataset_sha256'] == sha(DATA / f'{split}_tasks.jsonl')
    assert config['decoding'] == 'greedy' and config['max_trajectory'] == 6144 and config['max_turn_tokens'] == 512
    return {r['id']: r['metrics'] for r in rows}, config, sha(path)


def zero_variance(run):
    """Share of accepted training groups whose outcome rewards are all equal, by kind."""
    groups = defaultdict(list)
    for row in read_jsonl(run / 'rollouts.jsonl'):
        if row.get('accepted', True):
            groups[row['sample_group']].append(row['reward'])
    kinds = Counter()
    for rewards in groups.values():
        if max(rewards) - min(rewards) > 1e-9:
            kinds['informative'] += 1
        elif rewards[0] >= 1 - 1e-9:
            kinds['all_full_reward'] += 1
        elif abs(rewards[0]) < 1e-9:
            kinds['all_zero'] += 1
        elif rewards[0] <= -2:
            kinds['all_penalty'] += 1
        else:
            kinds['other_tie'] += 1
    return dict(groups=len(groups), **{k: v / len(groups) for k, v in sorted(kinds.items())})


def check_9b_provenance(eval_configs):
    """Adapted 9B evaluations must use the adapter trained on Qwen3.5-9B from the same 9B SFT start."""
    sft = json.loads((TRAINED_9B['rationale-sft'] / 'summary.json').read_text())
    assert sft['teacher'] == 'Qwen3.5-9B' and sft['data'] == 'data/v5/sft_err.jsonl'
    assert eval_configs['rationale-sft']['adapter_sha256'] == sft['adapter_sha256']
    assert eval_configs['base']['adapter_sha256'] == 'base'
    budgets = set()
    for name in ('grpo', 'vcr', 'vcr_mul'):
        summary = json.loads((TRAINED_9B[name] / 'summary.json').read_text())
        spec = summary['spec']
        assert Path(spec['base_model']).name == 'Qwen3.5-9B', name
        assert summary['initial_adapter_sha256'] == summary['reference_adapter_sha256'] == sft['adapter_sha256'], name
        assert summary['reference_frozen_verified'] and summary['global_step'] == spec['steps'] == 2048, name
        assert eval_configs[name]['adapter_sha256'] == summary['adapter_sha256'], name
        budgets.add((spec['learning_rate'], spec['beta'], spec['rank'], spec['dataset'], summary['dataset_sha256'],
                     summary['sampled_groups'], summary['accepted_groups']))
    assert len(budgets) == 1, 'The 9B RL arms must share data, budget and optimisation settings'
    return dict(sft_adapter_sha256=sft['adapter_sha256'], base_model='Qwen3.5-9B',
                note='Evaluation configs do not record the base model; the unadapted 9B row is identified by directory only.')


def main():
    result = {'provenance': 'Recorded experiments before source extraction; single training seed.', 'splits': {}}
    configs_2b = []
    for split in ('dev2', 'test'):
        tasks = read_jsonl(DATA / f'{split}_tasks.jsonl')
        expected = {t['id'] for t in tasks if t['domain'] == 'general'}
        assert len(expected) == 512
        models, hashes, configs = {}, {}, []
        for model in ('grpo', 'igpo', 'vcr'):
            models[model], config, hashes[model] = load_agent(RECORDED[model], split, expected)
            configs.append(config)
        for key in PROTOCOL_KEYS:
            assert len({c[key] for c in configs}) == 1, key
        configs_2b += configs
        ids = sorted(expected)
        result['splits'][split] = dict(
            tasks=len(ids), dataset_sha256=sha(DATA / f'{split}_tasks.jsonl'), predictions_sha256=hashes,
            stats={m: stats(v, ids) for m, v in models.items()},
            vcr_minus_grpo={
                'em': paired(models['vcr'], models['grpo'], ids, em),
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

    # Qwen3.5-9B: same tasks, decoding, budgets and evaluation protocol hashes as the 2B comparison.
    scale = dict(splits={})
    for split in ('dev2', 'test'):
        expected = {t['id'] for t in read_jsonl(DATA / f'{split}_tasks.jsonl') if t['domain'] == 'general'}
        models, hashes, configs = {}, {}, {}
        for name, root in RECORDED_9B.items():
            models[name], configs[name], hashes[name] = load_agent(root, split, expected)
        for key in PROTOCOL_KEYS:
            assert len({c[key] for c in configs.values()} | {c[key] for c in configs_2b if c['split'] == split}) == 1, key
        scale['provenance'] = check_9b_provenance(configs)
        ids = sorted(expected)
        metric = dict(em=em, early_stop=early, full_evidence=full)
        scale['splits'][split] = dict(
            predictions_sha256=hashes, stats={m: stats(v, ids) for m, v in models.items()},
            paired={f'{a}_minus_{b}': {k: paired(models[a], models[b], ids, f) for k, f in metric.items()}
                    for a, b in (('rationale-sft', 'base'), ('grpo', 'rationale-sft'),
                                 ('vcr', 'grpo'), ('vcr_mul', 'grpo'), ('vcr_mul', 'vcr'))})
    scale['zero_variance_training_groups'] = {
        **{f'2b-{k}': zero_variance(v) for k, v in TRAINED_2B.items()},
        **{f'9b-{k}': zero_variance(v) for k, v in TRAINED_9B.items() if k != 'rationale-sft'}}
    scale['vcr_mul'] = dict(
        advantage='A_GRPO * clip(exp(sign(A_GRPO) * c_t), 0.8, 1.2), claims verified exactly as in additive VCR',
        process=('Designed after the additive 9B VCR test result was inspected; no offline gate was run '
                 'and the 4-step probe was skipped. Its test EM gain is recorded as an observation, not a significant result.'))
    result['scale_9b'] = scale
    result['limitations'] = [
        'The test split was reused and inspected during iterative research; it is not a fresh blind test.',
        'Bootstrap intervals cover task sampling, not training-seed uncertainty.',
        'The 2B rationale-SFT checkpoint has dev2 evaluation but no complete recorded test evaluation.',
        'The 9B multiplicative-VCR arm was designed after inspecting the additive 9B VCR test result.',
    ]
    path = CODE / 'results/current.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('Recorded test results (not a retraining of this checkout):')
    for label, table in (('2B', result['splits']['test']['stats']), ('9B', scale['splits']['test']['stats'])):
        for name, value in table.items():
            print(f'{label} {name:13s} EM={100*value["em"]:.2f}% F1={100*value["f1"]:.2f}% '
                  f'full={100*value["full_rate"]:.2f}% early_stop={100*value["early_stop"]:.2f}%')
    for split in ('dev2', 'test'):
        p = scale['splits'][split]['paired']['vcr_mul_minus_grpo']
        print(f'9B vcr_mul - grpo {split}: EM {100*p["em"][0]:+.2f} [{100*p["em"][1]:+.2f}, {100*p["em"][2]:+.2f}]'
              f'  early stop {100*p["early_stop"][0]:+.2f} [{100*p["early_stop"][1]:+.2f}, {100*p["early_stop"][2]:+.2f}]')
    print('Wrote', path)


if __name__ == '__main__':
    main()
