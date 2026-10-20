"""Assistant-only LoRA SFT for the current 2B evidence agent."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from resources import ROOT, admit, lock_run, memory_record, read_jsonl, setup_paths

from evisearch.artifacts import CODE, RUNS, FROZEN_SFT, run_name, source_files, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen3.5-2B', type=run_name, help='base model under models/')
    parser.add_argument('--data', default=str(FROZEN_SFT))
    parser.add_argument('--name', type=run_name, help='run name; default sft')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cap-gib', type=float, default=8)
    args = parser.parse_args()
    out = RUNS / ((args.name or 'sft') + ('-probe' if args.probe else ''))
    out.mkdir(parents=True, exist_ok=True)
    lock = lock_run(out)
    if (out / 'summary.json').exists():
        print('Already complete:', out)
        return
    data = ROOT / args.data
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    assert data_sha == json.loads(data.with_suffix('.manifest.json').read_text())['sha256'], 'SFT data changed'
    student = json.loads((CODE / 'configs/sft-reference.json').read_text())
    lora = json.loads((CODE / 'configs/lora.json').read_text())
    identity = dict(model=args.model, data_sha256=data_sha, probe=args.probe,
                    source_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in source_files())).hexdigest())
    run_path = out / 'run.json'
    if run_path.exists():
        if not args.resume or json.loads(run_path.read_text()) != identity:
            raise ValueError('Existing SFT run needs matching --resume, code, model and data')
    elif args.resume:
        raise ValueError('Cannot resume without the original SFT run manifest')
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if checkpoint.parent != out.resolve() or not (checkpoint / 'trainer_state.json').is_file():
            raise ValueError('SFT checkpoint must belong to this run')
    run_path.write_text(json.dumps(identity, indent=2) + '\n')
    admit(args.gpu, args.cap_gib)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, set_seed
    from trl import SFTConfig, SFTTrainer
    from evisearch.model import tokenizer
    set_seed(42)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        ROOT / 'models' / args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 0}, output_loading_info=True)
    assert not (info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs')), info
    model = get_peft_model(model, LoraConfig(r=lora['r'], lora_alpha=lora['lora_alpha'], lora_dropout=0.0,
                                             task_type='CAUSAL_LM', target_modules=lora['target_modules']))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tok = tokenizer()
    rows = read_jsonl(data)
    if args.probe:
        rows = [max(rows, key=lambda r: r['tokens']), max(rows, key=lambda r: r['assistant_tokens'])] * 4
    dataset = Dataset.from_list([{k: r[k] for k in ('input_ids', 'labels', 'messages')} for r in rows])
    config = SFTConfig(output_dir=str(out), per_device_train_batch_size=1, bf16=True,
                       gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                       optim='adamw_torch', max_grad_norm=1.0, weight_decay=0.0, logging_steps=1,
                       save_steps=100, save_total_limit=2, report_to='tensorboard', logging_dir=str(out / 'tensorboard'),
                       seed=42, data_seed=42, dataloader_num_workers=0, dataloader_pin_memory=False,
                       learning_rate=student['learning_rate'], num_train_epochs=1, max_steps=2 if args.probe else -1,
                       gradient_accumulation_steps=student['gradient_accumulation_steps'],
                       warmup_ratio=student.get('warmup_ratio', 0.03), lr_scheduler_type='cosine', max_length=None,
                       assistant_only_loss=True, loss_type='chunked_nll', packing=False,
                       dataset_kwargs={'skip_prepare_dataset': True})
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tok)
    actual = json.loads(trainer.args.to_json_string())
    allowed = {'output_dir', 'logging_dir', 'max_steps', 'save_steps', 'save_total_limit'}
    diff = {k: [student.get(k), actual.get(k)] for k in student.keys() | actual.keys()
            if k not in allowed and student.get(k) != actual.get(k)}
    (out / 'config-diff.json').write_text(json.dumps(diff, indent=2) + '\n')

    from transformers import TrainerCallback

    class Guard(TrainerCallback):
        stopped_for_headroom = False

        def on_step_end(self, a, state, control, **kwargs):
            record = dict(step=state.global_step, **memory_record(args.gpu))
            with (out / 'memory.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if record['free_mib'] < 6144:
                self.stopped_for_headroom = True
                (out / 'paused.json').write_text(json.dumps(dict(step=state.global_step, reason='GPU headroom')))
                control.should_save = True
                control.should_training_stop = True

        def on_log(self, a, state, control, logs=None, **kwargs):
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=state.global_step, **(logs or {}))) + '\n')

    guard = Guard()
    trainer.add_callback(guard)
    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume)
    if guard.stopped_for_headroom:
        raise RuntimeError('GPU headroom below 6 GiB; resume from the latest checkpoint')
    (out / 'paused.json').unlink(missing_ok=True)
    trainer.save_model(str(out / 'final'))
    summary = dict(model=args.model, data=args.data, data_sha256=data_sha, steps=trainer.state.global_step,
                   seconds=time.time() - started, metrics=result.metrics, memory=memory_record(args.gpu),
                   adapter_sha256=hashlib.sha256((out / 'final/adapter_model.safetensors').read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
