"""Why do answers stay wrong when the evidence is already complete? CPU only, recorded predictions.

python -m evisearch.evaluation.answer_errors [--split test] [--examples 3]

Takes general-domain questions where the policy cited every annotated support document
(evidence_coverage >= 1) yet scored EM = 0, and classifies the submitted answer, in this order:

  insufficient    the policy declared 'insufficient evidence' although it held the full chain
  bridge_entity   the answer is an intermediate hop's gold sub-answer, not the final one
  near_miss       token F1 >= 0.5 with a gold alias: extra words, a longer phrase or a variant form
  from_distractor the answer text occurs only in documents outside the annotated support set
  wrong_span      the answer text occurs in the support documents, but is neither gold nor a bridge answer
  not_in_corpus   the answer occurs in no document of that question

Categories are assigned by that priority, so each question is counted once. Membership is tested on
normalised text (lowercase, punctuation and articles stripped), so it is a surface-form check, not
entailment. Writes results/answer-errors-<split>.json.
"""
import argparse
from collections import Counter
import json
import statistics

from resources import read_jsonl, setup_paths
from evisearch.artifacts import CODE, DATA, RECORDED, RECORDED_9B, sha
from evisearch.retrieval import answer_metrics, normalize

MODELS = (('2b-grpo', RECORDED, 'grpo'), ('2b-igpo', RECORDED, 'igpo'), ('2b-vcr', RECORDED, 'vcr'),
          ('9b-sft', RECORDED_9B, 'rationale-sft'),
          ('9b-grpo', RECORDED_9B, 'grpo'), ('9b-vcr', RECORDED_9B, 'vcr'),
          ('9b-vcr_mul', RECORDED_9B, 'vcr_mul'))
NEAR = 0.5


def load(root, split):
    path = root / split / 'predictions.jsonl'
    summary = json.loads((path.parent / 'summary.json').read_text())
    assert summary['predictions_sha256'] == sha(path), 'Predictions changed after evaluation'
    assert summary['config']['dataset_sha256'] == sha(DATA / f'{split}_tasks.jsonl')
    return [r for r in read_jsonl(path) if r['mode'] == 'agent' and r['domain'] == 'general']


def classify(task, answer):
    """Priority-ordered label for one wrong answer on a question whose evidence chain is complete."""
    predicted = normalize(answer)
    hops = task.get('decomposition') or []
    support = {d['id'] for d in task['documents'] if d['id'] in set(task['support_ids'])}
    text = {True: ' '.join(normalize(d['text']) for d in task['documents'] if d['id'] in support),
            False: ' '.join(normalize(d['text']) for d in task['documents'] if d['id'] not in support)}
    _, gold_f1 = answer_metrics(answer, task['answers'])
    if not predicted or 'insufficient evidence' in answer.lower():
        return 'insufficient', gold_f1
    # Intermediate hops only: the last hop's sub-answer is the final answer.
    if any(answer_metrics(answer, [hop['answer']])[1] >= NEAR for hop in hops[:-1]):
        return 'bridge_entity', gold_f1
    if gold_f1 >= NEAR:
        return 'near_miss', gold_f1
    found = [part for part, body in text.items() if predicted and predicted in body]
    if not found:
        return 'not_in_corpus', gold_f1
    return ('wrong_span' if True in found else 'from_distractor'), gold_f1


def analyse(rows, tasks, examples):
    complete = [r for r in rows if r['metrics']['evidence_coverage'] >= 1]
    wrong = [r for r in complete if r['metrics']['em'] < 1]
    labels, f1s, shown = Counter(), {}, {}
    for r in wrong:
        task = tasks[r['id']]
        label, gold_f1 = classify(task, r['metrics']['answer'] or '')
        labels[label] += 1
        f1s.setdefault(label, []).append(gold_f1)
        if len(shown.setdefault(label, [])) < examples:
            hops = task.get('decomposition') or []
            shown[label].append(dict(question=task['question'], gold=task['answers'],
                                     predicted=r['metrics']['answer'],
                                     hops=[f"{h['question']} -> {h['answer']}" for h in hops]))
    n = max(1, len(wrong))
    return dict(questions=len(rows), complete_evidence=len(complete),
                em_given_complete=statistics.fmean(r['metrics']['em'] for r in complete) if complete else None,
                wrong_with_complete_evidence=len(wrong),
                share_of_all_questions=len(wrong) / max(1, len(rows)),
                categories={k: dict(count=v, share=v / n, mean_gold_f1=statistics.fmean(f1s[k]))
                            for k, v in labels.most_common()},
                examples=shown)


def contained(answer, aliases):
    """Gold alias inside the answer, or the answer inside a gold alias, after normalisation."""
    predicted = normalize(answer)
    return float(bool(predicted) and any(gold and (gold in predicted or predicted in gold)
                                         for gold in map(normalize, aliases)))


def sensitivity(tasks, split, pairs):
    """Does a claim survive if answer-form mismatches are forgiven? Diagnostic only: EM stays the metric."""
    from evisearch.evaluation.compare import paired
    scored = {}
    for name, root, key in MODELS:
        scored[name] = {r['id']: dict(em=r['metrics']['em'],
                                      contained=max(r['metrics']['em'],
                                                    contained(r['metrics']['answer'] or '', tasks[r['id']]['answers'])))
                        for r in load(root[key], split)}
    rates = {name: dict(em=statistics.fmean(v['em'] for v in d.values()),
                        contained=statistics.fmean(v['contained'] for v in d.values()),
                        form_mismatch=statistics.fmean(v['contained'] - v['em'] for v in d.values()))
             for name, d in scored.items()}
    comparisons = {}
    for a, b in pairs:
        if a not in scored or b not in scored:
            continue
        ids = sorted(scored[a])
        comparisons[f'{a}_minus_{b}'] = dict(
            em=paired(scored[a], scored[b], ids, lambda x: x['em']),
            contained=paired(scored[a], scored[b], ids, lambda x: x['contained']))
    return dict(rates=rates, paired=comparisons,
                criterion='contained: EM, or a gold alias and the answer contain one another after normalisation')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', default='test', choices=('dev2', 'test'))
    parser.add_argument('--examples', type=int, default=3)
    args = parser.parse_args()
    setup_paths()
    tasks = {t['id']: t for t in read_jsonl(DATA / f'{args.split}_tasks.jsonl')}
    result = {name: analyse(load(root[key], args.split), tasks, args.examples) for name, root, key in MODELS}
    result['scoring_sensitivity'] = sensitivity(tasks, args.split, (
        ('9b-vcr_mul', '9b-grpo'), ('9b-vcr', '9b-grpo'), ('2b-vcr', '2b-grpo'), ('2b-vcr', '2b-igpo')))
    result['note'] = ('Surface-form classification on normalised text, not entailment. '
                      'Complete evidence means every annotated support document was cited; '
                      'the subsets differ between models, so counts are not paired.')
    path = CODE / f'results/answer-errors-{args.split}.json'
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    for name, _, _ in MODELS:
        v = result[name]
        if not isinstance(v, dict) or 'categories' not in v:
            continue
        print(f"{name:11s} 证据齐全 {v['complete_evidence']:3d} 题，其中答错 {v['wrong_with_complete_evidence']:3d} "
              f"(占全部题 {100*v['share_of_all_questions']:.1f}%)  " +
              '  '.join(f"{k} {100*x['share']:.0f}%" for k, x in v['categories'].items()))
    rates = result['scoring_sensitivity']['rates']
    print('\n答案形式造成的 EM 损失（含金标 EM − EM）：')
    for name in rates:
        r = rates[name]
        print(f"  {name:11s} EM {100*r['em']:5.2f}  含金标EM {100*r['contained']:5.2f}  形式损失 {100*r['form_mismatch']:5.2f}")
    print('\n关键配对在两种口径下：')
    for key, v in result['scoring_sensitivity']['paired'].items():
        print(f"  {key:24s} EM {100*v['em'][0]:+.2f} [{100*v['em'][1]:+.2f}, {100*v['em'][2]:+.2f}]"
              f"   含金标EM {100*v['contained'][0]:+.2f} [{100*v['contained'][1]:+.2f}, {100*v['contained'][2]:+.2f}]")
    print('Wrote', path)


if __name__ == '__main__':
    main()
