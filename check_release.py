"""CPU regression checks for the current publication's signal and entrypoints."""
import contextlib
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from resources import ROOT, setup_paths
from evisearch.artifacts import CODE, run_name


def check_frozen_data():
    from itertools import combinations
    from resources import read_jsonl
    from evisearch.artifacts import DATA, PROTOCOL, SEEDS, FROZEN_SFT, sha
    from evisearch.retrieval import normalize
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name in ('rl.jsonl', 'rl_tasks.jsonl', 'dev_tasks.jsonl', 'test_tasks.jsonl'):
        assert sha(DATA / name) == manifest['hashes'][name], name
    assert sha(DATA / 'dev2_tasks.jsonl') == json.loads((PROTOCOL / 'dev2_manifest.json').read_text())['dataset_sha256']
    assert sha(FROZEN_SFT) == json.loads(FROZEN_SFT.with_suffix('.manifest.json').read_text())['sha256']
    parents, count = set(), 0
    with FROZEN_SFT.open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            if row.get('parent_id'):
                parents.add(row['parent_id'])
            assert len(row['input_ids']) == len(row['labels'])
            assert any(label != -100 for label in row['labels'])
            assert all(label == -100 or label == token for token, label in zip(row['input_ids'], row['labels']))
    assert count == 6000
    splits = {name: read_jsonl(DATA / f'{name}_tasks.jsonl') for name in ('rl', 'dev', 'dev2', 'test')}
    assert {k: len(v) for k, v in splits.items()} == dict(rl=1024, dev=128, dev2=512, test=512)
    splits['sft'] = [t for t in read_jsonl(SEEDS / 'sft_pool.jsonl') if t['id'] in parents]
    assert {t['id'] for t in splits['sft']} == parents
    for (left, a), (right, b) in combinations(splits.items(), 2):
        for key in ('id', 'question'):
            value = (lambda t: normalize(t[key])) if key == 'question' else (lambda t: t[key])
            assert not {value(t) for t in a} & {value(t) for t in b}, (left, right, key)
        assert not {p['id'] for t in a for p in t.get('decomposition', [])} & {
            p['id'] for t in b for p in t.get('decomposition', [])}, (left, right, 'components')


def check_vcr():
    from evisearch.data.teacher import contract_task
    from evisearch.training.vcr import AgentTrainer
    from evisearch.training.advantages import normalize_group
    from trl import GRPOTrainer

    task = contract_task()
    task['decomposition'] = [dict(answer='Example University', support_id='d1')]
    messages = [dict(role='assistant', content='"Journal Z" answers step 1: Example University.'),
                dict(role='tool', content='not a training token'),
                dict(role='assistant', content='"Unrelated Institute" answers step 1: Example University.')]
    trainer = object.__new__(AgentTrainer)
    trainer.raw_rollout = (None, None, [[1, 1, 0, 1, 1]], [messages])
    trainer.records = [dict(id=task['id'])]
    trainer._tasks = [task]
    trainer._metrics = {'train': defaultdict(list)}
    result = dict(completion_mask=torch.ones(1, 5), tool_mask=torch.tensor([[1, 1, 0, 1, 1]]),
                  advantages=torch.tensor([0.0]))
    trainer._apply_vcr(result)
    assert result['advantages'].tolist() == [[0.5, 0.5, 0.0, -0.5, -0.5]]
    assert normalize_group([0, 0, 0, 0]) == [0, 0, 0, 0]

    # Multiplicative VCR: bounded rescaling that keeps the sign of A_GRPO and adds nothing when A_GRPO = 0.
    trainer.algorithm = 'vcr_mul'
    for sequence, expected in ((0.5, [0.6, 0.6, 0.0, 0.4, 0.4]), (-0.5, [-0.4, -0.4, 0.0, -0.6, -0.6]),
                               (0.0, [0.0, 0.0, 0.0, 0.0, 0.0])):
        result = dict(completion_mask=torch.ones(1, 5), tool_mask=torch.tensor([[1, 1, 0, 1, 1]]),
                      advantages=torch.tensor([sequence]))
        trainer._apply_vcr(result)
        assert torch.allclose(result['advantages'], torch.tensor([expected])), (sequence, result['advantages'])
    del trainer.algorithm

    # Use the actual installed TRL loss, so a mask regression changes gradients.
    for advantages in (torch.tensor([1., -1.]), torch.tensor([[1., 0., -.5], [-1., 0., 0.]])):
        values = torch.tensor([[.3, 7., -.1], [.1, -8., .2]], requires_grad=True)
        native = object.__new__(GRPOTrainer)
        native.model = SimpleNamespace(training=True)
        native._get_per_token_logps_and_entropies = lambda *a, **k: (values, torch.ones_like(values), None)
        native.aux_loss_enabled = False
        native.top_entropy_quantile = 1.
        native.off_policy_mask_threshold = None
        native.importance_sampling_level = 'token'
        native.beta, native.loss_type = 0., 'grpo'
        native.epsilon_low = native.epsilon_high = .2
        native.args = SimpleNamespace(delta=None, steps_per_generation=1)
        native.current_gradient_accumulation_steps = 1
        native.use_vllm = native._entropy_bonus_enabled = False
        native._metrics = {'train': defaultdict(list)}
        native.accelerator = SimpleNamespace(num_processes=1, gather=lambda x: x, reduce=lambda x, **k: x)
        mask = torch.tensor([[1, 0, 1], [1, 0, 0]])
        inputs = dict(prompt_ids=torch.ones(2, 1, dtype=torch.long), prompt_mask=torch.ones(2, 1),
                      completion_ids=torch.ones(2, 3, dtype=torch.long), completion_mask=torch.ones(2, 3),
                      tool_mask=mask, advantages=advantages, old_per_token_logps=torch.zeros_like(values),
                      num_items_in_batch=mask.sum())
        loss = native._compute_loss(native.model, inputs)
        loss.backward()
        assert values.grad[mask == 0].count_nonzero() == 0
        assert values.grad.count_nonzero() > 0


def check_comparison(root):
    from evisearch.evaluation import behavior, compare
    data, runs = root / 'data', root / 'runs'
    data.mkdir()
    config = dict(decoding='greedy', max_trajectory=6144, max_turn_tokens=512,
                  source_sha256='same-code', protocol_sha256='same-protocol')
    expected = [{'id': 'a'}, {'id': 'b'}]
    for split in ('dev2', 'test'):
        tasks = data / f'{split}_tasks.jsonl'
        tasks.write_text(''.join(json.dumps(t) + '\n' for t in expected))
        for model in ('target', 'baseline'):
            out = runs / 'eval' / model / split
            out.mkdir(parents=True)
            cfg = dict(config, dataset_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest())
            (out / 'config.json').write_text(json.dumps(cfg))
            rows = [dict(id=t['id'], mode='agent', metrics=dict(em=0., f1=0., evidence_coverage=0.,
                    format_valid=True, search_calls=3)) for t in expected]
            pred = out / 'predictions.jsonl'
            pred.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (out / 'summary.json').write_text(json.dumps(dict(config=cfg,
                predictions_sha256=hashlib.sha256(pred.read_bytes()).hexdigest())))
    with patch('evisearch.artifacts.RUNS', runs), patch.object(behavior, 'DATA', data), patch.object(compare, 'RUNS', runs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            compare.main('target', 'baseline')
        assert 'n/a' in output.getvalue(), 'Zero complete-evidence cases must still print'
        # Even with a consistent summary hash, an incomplete shared subset must fail.
        pred = runs / 'eval/target/test/predictions.jsonl'
        pred.write_text(pred.read_text().splitlines()[0] + '\n')
        summary_path = pred.parent / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['predictions_sha256'] = hashlib.sha256(pred.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary))
        try:
            behavior.predictions('target', 'test')
        except AssertionError as exc:
            assert 'Incomplete' in str(exc)
        else:
            raise AssertionError('Partial task set was accepted')


def check_sft_resume(root):
    """Exercise main() with a fake GPU/trainer; no real model or allocation."""
    from evisearch import sft
    import datasets
    import peft
    import transformers
    import trl

    data = root / 'sft.jsonl'
    row = dict(input_ids=[1], labels=[1], messages=[], tokens=1, assistant_tokens=1)
    data.write_text(json.dumps(row) + '\n')
    data.with_suffix('.manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(data.read_bytes()).hexdigest())))
    runs = root / 'sft-runs'
    fake_model = SimpleNamespace(config=SimpleNamespace(), enable_input_require_grads=lambda: None,
                                  gradient_checkpointing_enable=lambda **kw: None)
    free = [5000]

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.args = kwargs['args']
            self.state = SimpleNamespace(global_step=2)

        def add_callback(self, callback):
            self.callback = callback

        def train(self, resume_from_checkpoint=None):
            out = Path(self.args.output_dir)
            checkpoint = out / 'checkpoint-2'
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / 'trainer_state.json').write_text('{"global_step": 2}')
            self.callback.on_step_end(None, self.state, SimpleNamespace())
            return SimpleNamespace(metrics={})

        def save_model(self, path):
            final = Path(path)
            final.mkdir()
            (final / 'adapter_model.safetensors').write_bytes(b'test-not-a-real-adapter')

    def config(**kwargs):
        return SimpleNamespace(**kwargs, to_json_string=lambda: json.dumps(kwargs))

    argv = ['sft', '--data', str(data)]
    with contextlib.ExitStack() as stack:
        for target, attr, value in (
            (sft, 'RUNS', runs), (sft, 'admit', lambda *a, **k: None),
            (sft, 'memory_record', lambda *a, **k: dict(free_mib=free[0])),
            (trl, 'SFTConfig', config), (trl, 'SFTTrainer', FakeTrainer),
            (datasets.Dataset, 'from_list', lambda rows: rows),
            (transformers.Qwen3_5ForCausalLM, 'from_pretrained', lambda *a, **k: (fake_model, {})),
            (peft, 'get_peft_model', lambda model, config: model),
        ):
            stack.enter_context(patch.object(target, attr, value))
        stack.enter_context(patch('evisearch.model.tokenizer', return_value=None))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        with patch.object(sys, 'argv', argv):
            try:
                sft.main()
            except RuntimeError as exc:
                assert 'headroom' in str(exc)
            else:
                raise AssertionError('Low memory must pause')
        out = runs / 'sft'
        assert (out / 'paused.json').exists() and not (out / 'final').exists()
        checkpoint = str(out / 'checkpoint-2')
        with patch.object(sys, 'argv', argv + ['--resume', checkpoint, '--model', 'DifferentModel']):
            try:
                sft.main()
            except ValueError as exc:
                assert 'matching' in str(exc)
            else:
                raise AssertionError('Changed-model resume accepted')
        free[0] = 20000
        with patch.object(sys, 'argv', argv + ['--resume', checkpoint]):
            sft.main()
        assert (out / 'summary.json').exists() and (out / 'final').exists()
        assert not (out / 'paused.json').exists()


def main():
    setup_paths()
    torch.set_num_threads(2)
    for invalid in ('../old-run', '/tmp/escape', '.', '..', 'a/b'):
        try:
            run_name(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
    assert run_name('vcr-42') == 'vcr-42'
    from evisearch import train
    assert train.specification('vcr_mul')['name'] == 'vcr_mul-42', 'Default 2B run names must not change'
    spec = train.specification('vcr_mul', base='Qwen3.5-9B', sft='sft-9b')
    assert spec['name'] == 'vcr_mul-Qwen3.5-9B-42' and spec['initial'] == spec['reference'] == str(train.RUNS / 'sft-9b/final')
    check_frozen_data()
    check_vcr()
    with TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
        check_comparison(Path(tmp))
        check_sft_resume(Path(tmp))
    assert not any(p.name.startswith('RESULTS') for p in CODE.rglob('*.md'))
    print('PASS: frozen splits/data, VCR and native loss gradients, full-denominator comparisons, path isolation, SFT pause/resume and identity')


if __name__ == '__main__':
    main()
