"""Same-SFT GRPO / IGPO / VCR with native TRL tools and frozen-reference audits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from resources import ROOT, admit, lock_run, memory_record, read_jsonl, setup_paths
from evisearch.artifacts import PROTOCOL, DATA, RUNS, sha, CODE, source_files

setup_paths()
ARMS = ('grpo', 'igpo', 'vcr')


def specification(arm, seed=42, probe=False):
    assert arm in ARMS
    initial = RUNS / 'sft' / 'final'
    return dict(name=f'{arm}-{seed}' + ('-probe' if probe else ''), sft=False,
                algorithm=arm, initial=str(initial), reference=str(initial),
                learning_rate=1e-5, beta=0.02, rank=8, steps=4 if probe else 2048,
                dataset=str(DATA / 'rl.jsonl'))


def output_dir(spec):
    return RUNS / spec['name']


def code_files(algorithm):
    return source_files()


def code_hash(algorithm):
    return hashlib.sha256(b''.join(p.read_bytes() for p in code_files(algorithm))).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=ARMS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cap-gib', type=float, default=8)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    spec = specification(args.arm, args.seed, args.probe)
    args.algorithm = spec['algorithm']
    out = output_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    lock = lock_run(out)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed run already exists')
    source_hash = code_hash(args.algorithm)
    data_hash = sha(PROTOCOL / 'manifest.json')
    checks = json.loads((PROTOCOL / 'checks.json').read_text())
    assert checks['passed'] and checks['manifest_sha256'] == data_hash, 'Frozen data checks required'
    assert sha(DATA / 'manifest.json') == json.loads((PROTOCOL / 'manifest.json').read_text())['v3_manifest_sha256']
    data_manifest = json.loads((DATA / 'manifest.json').read_text())
    expected = data_manifest['hashes'][Path(spec['dataset']).name]
    assert sha(Path(spec['dataset'])) == expected
    initial_hash = sha(Path(spec['initial']) / 'adapter_model.safetensors') if spec['initial'] else None
    reference_hash = sha(Path(spec['reference']) / 'adapter_model.safetensors') if spec['reference'] else None
    key = dict(algorithm=args.algorithm, arm=args.arm, seed=args.seed, probe=args.probe,
               spec=spec, source_hash=source_hash, data_hash=data_hash, dataset_sha256=expected,
               initial_adapter_sha256=initial_hash, reference_adapter_sha256=reference_hash)
    if (out / 'run.json').exists():
        if not args.resume or json.loads((out / 'run.json').read_text()) != key:
            raise ValueError('Existing run needs matching --resume, code and dataset')
    (out / 'run.json').write_text(json.dumps(key, indent=2) + '\n')
    base = CODE
    snapshot = {}
    for path in code_files(args.algorithm):
        relative = path.relative_to(base)
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        if destination.exists():
            assert destination.read_bytes() == content, 'Training source snapshot changed'
        else:
            destination.write_bytes(content)
        snapshot[str(relative)] = hashlib.sha256(content).hexdigest()
    (out / 'source/manifest.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    resume_step, previous_gradient_steps = 0, 0
    if args.resume:
        assert args.resume.resolve().parent == out.resolve()
        state = json.loads((args.resume / 'trainer_state.json').read_text())
        step = resume_step = state['global_step']
        previous_gradient_steps = len({r['step'] for r in state['log_history']
                                       if r.get('grad_norm', 0) > 0 and r['step'] <= step})
        if args.algorithm != 'sft' and step % 2:
            raise ValueError('RL must resume at a complete two-update rollout boundary; select the previous even checkpoint')
        for name in ('metrics.jsonl', 'memory.jsonl', 'rollouts.jsonl', 'diagnostics.jsonl'):
            path = out / name
            if path.exists():
                path.with_name(name + f'.before-resume-{time.time_ns()}').write_bytes(path.read_bytes())
                rows = [r for r in read_jsonl(path, repair_tail=True) if r['step'] < step or
                        (name != 'rollouts.jsonl' and r['step'] == step)]
                path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    admit(args.gpu, args.cap_gib)
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig
    from evisearch.model import load_model, tokenizer, restore_policy
    from evisearch.environment import EvidenceEnv, MAX_TURNS
    from evisearch.training.trainer import AgentTrainer as GRPOTrainer, load_reference
    from evisearch.training.vcr import AgentTrainer as VCRTrainer
    AgentTrainer = VCRTrainer if args.algorithm == 'vcr' else GRPOTrainer

    set_seed(args.seed)
    adapter = Path(spec['initial']) if spec['initial'] else None
    model = load_model(adapter)
    tok = tokenizer()
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    class Audit(TrainerCallback):
        nonzero_gradient_steps = 0
        stopped_for_headroom = False

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            norms = [p.grad.detach().float().norm() for p in model.parameters() if p.requires_grad and p.grad is not None]
            norm = torch.stack(norms).norm().item() if norms else 0.0
            if not math.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            self.nonzero_gradient_steps += norm > 0

        def on_step_end(self, args, state, control, **kwargs):
            record = dict(step=state.global_step, **memory_record(gpu=args_gpu))
            with (out / 'memory.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if record['free_mib'] < 6144:
                self.stopped_for_headroom = True
                control.should_save = True
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=state.global_step, **(logs or {}))) + '\n')
            if not spec['sft'] and logs and 'grad_norm' in logs:
                outcomes = [r['outcome_reward'] for r in trainer.records]
                totals = getattr(trainer, 'kl_totals', [0.0, 0, 0.0, 0, 0.0])
                record = dict(step=state.global_step, sample_group=trainer.sampling_groups,
                    informative=max(outcomes) - min(outcomes) >= 1e-12, grad_norm=logs['grad_norm'],
                    kl_token_sum=totals[0], kl_tokens=totals[1], kl_sequence_mean_sum=totals[2], sequences=totals[3],
                    mean_token_kl=totals[0] / max(1, totals[1]), beta=spec['beta'],
                    # Native TRL logs a token mean per microbatch, then averages
                    # microbatches. With the frozen batch size 1 this is the sequence mean.
                    mean_update_kl=totals[4] / max(1, totals[3]),
                    kl_objective_contribution=spec['beta'] * totals[4] / max(1, totals[3]),
                    kl_scope='Actual update tokens, including both passes; prompt/tool/truncated masks excluded')
                with (out / 'diagnostics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                trainer.kl_totals = [0.0, 0, 0.0, 0, 0.0]

    args_gpu = args.gpu
    audit = Audit()
    common = dict(output_dir=str(out), per_device_train_batch_size=1, bf16=True,
                  gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                  optim='adamw_torch', max_grad_norm=1.0, weight_decay=0.0,
                  logging_steps=1, save_steps=100, save_total_limit=2, report_to='tensorboard',
                  logging_dir=str(out / 'tensorboard'), seed=args.seed, data_seed=42,
                  dataloader_num_workers=0, dataloader_pin_memory=False)
    rows = [json.loads(line) for line in Path(spec['dataset']).open()]
    dataset = Dataset.from_list(rows)
    # Two optimization passes per rollout group make policy-ratio clipping measurable.
    # Every main arm keeps 1,024 groups x 4 rollouts and 2,048 optimizer updates.
    config = GRPOConfig(**common, learning_rate=spec['learning_rate'], max_steps=spec['steps'],
                        gradient_accumulation_steps=4, num_generations=4, generation_batch_size=4,
                        max_completion_length=6144, max_tool_calling_iterations=MAX_TURNS - 1, temperature=1.0, top_p=0.95,
                        beta=spec['beta'], epsilon=0.2, epsilon_high=0.2,
                        loss_type='grpo', importance_sampling_level='token',
                        scale_rewards='group', num_iterations=2, use_vllm=False, use_liger_kernel=True,
                        shuffle_dataset=False,
                        generation_kwargs={'eos_token_id': tok.eos_token_id, 'use_cache': True, 'max_new_tokens': 512},
                        chat_template_kwargs={'enable_thinking': False},
                        mask_truncated_completions=True, log_completions=False)
    trainer = AgentTrainer(algorithm=args.algorithm, audit_dir=out, model=model, args=config,
                           train_dataset=dataset, environment_factory=EvidenceEnv,
                           processing_class=tok, callbacks=[audit])
    reference_check = load_reference(model, spec['reference'])
    reference_check['source_adapter_sha256'] = reference_hash
    (out / 'reference-check.json').write_text(json.dumps(reference_check, indent=2) + '\n')
    reference_initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.ref.' in n}
    (out / 'trainer_config.json').write_text(trainer.args.to_json_string())
    baseline = json.loads((CODE / 'configs/rl-reference.json').read_text())
    actual = json.loads(trainer.args.to_json_string())
    allowed = {'output_dir', 'logging_dir', 'seed', 'max_steps'}
    unexpected = {k: [baseline.get(k), actual.get(k)] for k in baseline.keys() | actual.keys()
                  if k not in allowed and baseline.get(k) != actual.get(k)}
    assert not unexpected, f'Unplanned training configuration differences: {unexpected}'
    (out / 'config-diff.json').write_text(json.dumps({k: [baseline.get(k), actual.get(k)]
        for k in allowed if baseline.get(k) != actual.get(k)}, indent=2) + '\n')
    if args.resume:
        restore_policy(model, args.resume)
        if args.algorithm != 'sft':
            saved = read_jsonl(out / 'rollouts.jsonl')
            groups = {r['sample_group']: r for r in saved}
            trainer.sampling_groups = max(groups, default=0)
            trainer.accepted_groups = sum(r['accepted'] for r in groups.values())
            trainer.dapo_cursor = sum(not r['accepted'] for r in groups.values())
    segments_path = out / 'training_segments.jsonl'
    previous_segments = read_jsonl(segments_path, repair_tail=True) if segments_path.exists() else []
    previous_elapsed = sum(r['elapsed_seconds'] for r in previous_segments)
    started = time.time()
    segment = dict(started_unix=started, start_step=resume_step, gpu=args.gpu,
                   resumed_from=str(args.resume) if args.resume else None, status='interrupted')
    try:
        result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        if audit.stopped_for_headroom:
            segment['status'] = 'paused'
            (out / 'paused.json').write_text(json.dumps({'step': trainer.state.global_step, 'reason': 'GPU headroom'}))
            return
        trainer.save_model(str(out / 'final'))
        trainer.save_state()
        changes = {n: (p.detach().cpu().float() - initial[n].float()).norm().item()
                   for n, p in model.named_parameters() if n in initial}
        reference_displacement = sum((p.detach().cpu().float() - reference_initial[n.replace('.default.', '.ref.')].float()).square().sum().item()
                                     for n, p in model.named_parameters() if n in initial) if reference_initial else None
        changed = any(changes.values())
        assert all(torch.equal(p.detach().cpu(), reference_initial[n])
                   for n, p in model.named_parameters() if n in reference_initial), 'Frozen reference changed'
        gradient_steps = previous_gradient_steps + audit.nonzero_gradient_steps
        if not args.probe or args.algorithm == 'sft':
            assert changed and gradient_steps > 0
        segment.update(status='completed', elapsed_seconds=time.time() - started)
        summary = dict(key, global_step=trainer.state.global_step,
                       changed_tensors=sum(v > 0 for v in changes.values()),
                       nonzero_gradient_steps=gradient_steps,
                       elapsed_seconds=previous_elapsed + segment['elapsed_seconds'],
                       last_segment_seconds=segment['elapsed_seconds'],
                       training_segments=len(previous_segments) + 1,
                       policy_changed=changed, allocator_cap_gib=args.cap_gib,
                       adapter_displacement_l2=math.sqrt(sum(v * v for v in changes.values())),
                       reference_adapter_displacement_l2=math.sqrt(reference_displacement) if reference_displacement is not None else None,
                       reference_frozen_verified=bool(reference_initial),
                       memory=memory_record(args.gpu), metrics=result.metrics,
                       adapter_sha256=hashlib.sha256((out / 'final/adapter_model.safetensors').read_bytes()).hexdigest())
        if (out / 'paused.json').exists():
            (out / 'paused.json').unlink()
        if not spec['sft']:
            summary['sampled_groups'] = trainer.sampling_groups
            summary['accepted_groups'] = trainer.accepted_groups
            assert trainer.accepted_groups * 2 == trainer.state.global_step
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        segment['status'] = 'failed'
        (out / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'error': str(exc),
                                                   'elapsed_seconds': time.time() - started}, indent=2))
        raise
    finally:
        segment.setdefault('elapsed_seconds', time.time() - started)
        segment.update(end_step=trainer.state.global_step,
                       nonzero_gradient_steps=audit.nonzero_gradient_steps)
        with segments_path.open('a') as stream:
            stream.write(json.dumps(segment) + '\n')


if __name__ == '__main__':
    main()
