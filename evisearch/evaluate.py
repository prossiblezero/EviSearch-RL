"""Evaluate the current agent with frozen data and greedy decoding."""
import argparse
import hashlib
import json
from pathlib import Path

from resources import ROOT, admit, lock_run, memory_record, read_jsonl, setup_paths
from evisearch.evaluation.rollout import episode, aggregate, MAX_TRAJECTORY, MAX_TURN
from evisearch.artifacts import DATA, PROTOCOL as OUT, RUNS, CODE, sha, run_name

def model_dir(name):
    return RUNS / name


def protocol_hash():
    return hashlib.sha256(b''.join((CODE / p).read_bytes() for p in (
        'evisearch/evaluation/rollout.py', 'evisearch/environment.py', 'evisearch/retrieval.py',
        'evisearch/model.py', 'evisearch/training/advantages.py'))).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=run_name)
    parser.add_argument('--split', choices=('dev2', 'dev', 'test', 'medical'), required=True)
    parser.add_argument('--modes', nargs='+', choices=('agent', 'static'), default=['agent'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cap-gib', type=float, default=8)
    args = parser.parse_args()
    setup_paths()
    checks = json.loads((OUT / 'checks.json').read_text())
    assert checks['passed'] and checks['manifest_sha256'] == sha(OUT / 'manifest.json')
    assert len(set(args.modes)) == len(args.modes)
    adapter = None if args.model == 'base' else model_dir(args.model) / 'final'
    if adapter:
        summary = json.loads((adapter.parent / 'summary.json').read_text())
        assert sha(adapter / 'adapter_model.safetensors') == summary['adapter_sha256']
    tasks_path = DATA / f'{args.split}_tasks.jsonl'
    tasks = read_jsonl(tasks_path)
    expected_hash = (json.loads((OUT / 'dev2_manifest.json').read_text())['dataset_sha256']
                     if args.split == 'dev2' else json.loads((DATA / 'manifest.json').read_text())['hashes'][tasks_path.name])
    assert sha(tasks_path) == expected_hash
    out = RUNS / 'eval' / args.model / args.split
    out.mkdir(parents=True, exist_ok=True)
    lock = lock_run(out)
    config = dict(model=args.model, adapter_sha256=sha(adapter / 'adapter_model.safetensors') if adapter else 'base',
        dataset_sha256=expected_hash, split=args.split, modes=args.modes, decoding='greedy',
        max_trajectory=MAX_TRAJECTORY, max_turn_tokens=MAX_TURN, expected_tasks=len(tasks),
        source_sha256=protocol_hash(), wrapper_sha256=sha(Path(__file__)),
        protocol_sha256=sha(OUT / 'protocol.json'), checkpoint_selection='Predeclared final update; no test-based selection')
    path = out / 'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, 'Evaluation configuration changed'
    path.write_text(json.dumps(config, indent=2) + '\n')
    if (out / 'summary.json').exists():
        result = json.loads((out / 'summary.json').read_text())
        assert result['predictions_sha256'] == sha(out / 'predictions.jsonl')
        print('Already complete:', out)
        return
    rows = read_jsonl(out / 'predictions.jsonl', repair_tail=True) if (out / 'predictions.jsonl').exists() else []
    done = {(r['mode'], r['id']) for r in rows}
    expected = {(mode, task['id']) for mode in args.modes for task in tasks}
    assert len(done) == len(rows) and done <= expected
    admit(args.gpu, args.cap_gib)
    from evisearch.model import load_model, tokenizer
    model, tok = load_model(adapter, trainable=False), tokenizer()
    model.eval()
    for mode in args.modes:
        for task in tasks:
            if (mode, task['id']) in done:
                continue
            if memory_record(args.gpu)['free_mib'] < 6144:
                raise RuntimeError('GPU headroom below 6 GiB; evaluation can resume per task')
            row = episode(model, tok, task, mode)
            rows.append(row)
            with (out / 'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps(dict(completed=len(rows), expected=len(expected), id=task['id'],
                                  em=row['metrics']['em'])), flush=True)
    assert {(r['mode'], r['id']) for r in rows} == expected
    summary = dict(config=config, results=aggregate(rows), memory=memory_record(args.gpu),
                   predictions_sha256=sha(out / 'predictions.jsonl'))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
