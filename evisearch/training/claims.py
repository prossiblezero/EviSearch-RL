"""Verifiable relevance claims (VCR): parse and verify the policy's own rationale sentences against MuSiQue gold.

Claims are the ERR sentence forms written before tool calls:
  '"<title>" answers step k (<sub-question>): <answer>.'   and   '"<title>" does not answer step k (<sub-question>)'
Verification (training-time only; uses gold decomposition and support documents):
  answers-claim is correct iff <title> is the gold document of step k and the stated answer has token F1 >= 0.5
  with the gold sub-answer; does-not-answer is correct iff <title> is not a gold document of any step.
  A claim naming a title that is not in the question's corpus is incorrect.
Usage (offline audit): python -m evisearch.training.claims SPLIT MODEL   -> runs/release/claims-<model>-<split>.json
"""
import json
import re
import statistics
import sys

from resources import read_jsonl, setup_paths
from evisearch.artifacts import DATA, RUNS

PATTERN = re.compile(r'"([^"]+)" (answers|does not answer) step (\d+)(?: \(([^)]*)\))?(?:: ([^."]*))?')


def verify(task, text):
    """List of (kind, correct) for every claim in one assistant turn's text."""
    from evisearch.retrieval import answer_metrics
    titles = {d['title']: d['id'] for d in task['documents']}
    gold = [p['support_id'] for p in task.get('decomposition') or []]
    out = []
    for title, kind, k, _, answer in PATTERN.findall(text or ''):
        k = int(k) - 1
        source = titles.get(title)
        if source is None or not 0 <= k < len(gold):
            out.append((kind, False))
        elif kind == 'answers':
            out.append((kind, source == gold[k] and answer_metrics(answer.strip(), [task['decomposition'][k]['answer']])[1] >= 0.5))
        else:
            out.append((kind, source not in gold))
    return out


def turn_rewards(task, completion):
    """Per assistant turn: sum of +1 (correct) / -1 (incorrect) claims, clipped to [-1, 1]; 0 when no claim."""
    rewards = []
    for message in completion:
        if message['role'] != 'assistant':
            continue
        claims = verify(task, message.get('content'))
        rewards.append(max(-1.0, min(1.0, float(sum(1 if ok else -1 for _, ok in claims)))) if claims else 0.0)
    return rewards


def main(split, model):
    setup_paths()
    tasks = {t['id']: t for t in read_jsonl(DATA / f'{split}_tasks.jsonl')}
    rows = [r for r in read_jsonl(RUNS / 'eval' / model / split / 'predictions.jsonl') if r['mode'] == 'agent']
    per = []
    for r in rows:
        t = tasks[r['id']]
        if t['domain'] != 'general':
            continue
        claims = [c for m in r['messages'][2:] if m['role'] == 'assistant' for c in verify(t, m.get('content'))]
        per.append(dict(em=r['metrics']['em'], full=r['metrics']['evidence_coverage'] >= 1, claims=len(claims),
                        false=sum(not ok for _, ok in claims), true=sum(ok for _, ok in claims)))
    mean = lambda v: statistics.fmean(v) if v else None
    any_false = [p for p in per if p['false']]
    clean = [p for p in per if p['claims'] and not p['false']]
    result = dict(questions=len(per), with_claims=sum(p['claims'] > 0 for p in per),
                  claim_accuracy=sum(p['true'] for p in per) / max(1, sum(p['claims'] for p in per)),
                  em_with_any_false_claim=mean([p['em'] for p in any_false]), n_any_false=len(any_false),
                  em_all_claims_true=mean([p['em'] for p in clean]), n_all_true=len(clean),
                  full_with_any_false_claim=mean([p['full'] for p in any_false]),
                  full_all_claims_true=mean([p['full'] for p in clean]))
    (RUNS / f'claims-{model}-{split}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
