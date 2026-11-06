"""Offline gate: do high-entropy assistant tokens sit on decision points? Inference only.

python -m evisearch.evaluation.entropy --gpu 1 --cap-gib 8

Re-renders recorded 2B GRPO training rollouts (last 1/8 of training) with the training chat template and tool
schemas, keeps only rollouts whose assistant-token count equals the recorded generated_tokens, and computes the
temperature-1 entropy of the final GRPO adapter at every assistant token. Tokens are labelled by character span:

  action    first token of each <tool_call> block and of the final JSON answer (the continue/stop commitment)
  query     search query value          doc     read doc_id / page value       answer  final answer string
  rationale assistant text outside tool calls and the final JSON               copy    passage_id / evidence_ids
  format    everything else (tags, JSON keys, end-of-turn tokens)

Gate (declared before running; research log, appendix after section 27), on informative groups only:
  1. enrichment: share of decision tokens (action, query, doc, answer, rationale) among the top-20% entropy
     tokens, divided by their share among all assistant tokens, >= 1.5
  2. stop/continue: median entropy of action tokens >= 80th percentile of all assistant-token entropies
Writes runs/release/entropy-gate/{tokens.jsonl, gate.json}.
"""
import argparse
from collections import defaultdict
import json
import math
import re
import statistics

from resources import admit, read_jsonl, setup_paths
from evisearch.artifacts import DATA, RUNS, TRAINED_2B

FIRST_STEP = 1792
DECISION = ('action', 'query', 'doc', 'answer', 'rationale')
PARAMETER = {('search', 'query'): 'query', ('read', 'doc_id'): 'doc', ('read', 'page'): 'doc',
             ('add_evidence', 'passage_id'): 'copy'}
OUT = RUNS / 'entropy-gate'
EMPTY_THINK = '<think>\n\n</think>\n\n'
early = lambda m: bool(m['evidence_coverage'] < 1 and m['search_calls'] < 3 and m['format_valid'])


def segments(mask):
    """Contiguous runs of assistant tokens as [start, end) token index pairs."""
    runs, start = [], None
    for i, value in enumerate(list(mask) + [0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    return runs


def spans(text, begin, end):
    """Labelled character spans inside one assistant segment, as (start, end, label), plus covered ranges."""
    part, out, covered = text[begin:end], [], []
    for block in re.finditer(r'<tool_call>.*?</tool_call>', part, re.S):
        b0 = begin + block.start()
        covered.append((b0, begin + block.end()))
        out.append((b0, b0 + len('<tool_call>'), 'action'))
        function = re.search(r'<function=(\w+)>', block.group(0))
        for parameter in re.finditer(r'<parameter=(\w+)>\n?(.*?)\n?</parameter>', block.group(0), re.S):
            label = PARAMETER.get((function.group(1) if function else '', parameter.group(1)), 'format')
            out.append((b0 + parameter.start(2), b0 + parameter.end(2), label))
    final = re.search(r'\{\s*"answer"', part)
    if final and not any(a <= begin + final.start() < b for a, b in covered):
        f0 = begin + final.start()
        close = part.rfind('}')
        covered.append((f0, begin + close + 1 if close >= final.start() else end))
        out.append((f0, f0 + 2, 'action'))
        answer = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', part[final.start():])
        if answer:
            out.append((f0 + answer.start(1), f0 + answer.end(1), 'answer'))
        cited = re.search(r'"evidence_ids"\s*:\s*\[(.*?)\]', part[final.start():], re.S)
        if cited:
            out.append((f0 + cited.start(1), f0 + cited.end(1), 'copy'))
    return out, covered


def label_tokens(text, offsets, mask):
    labels = [None] * len(offsets)
    for t0, t1 in segments(mask):
        begin, end = offsets[t0][0], offsets[t1 - 1][1]
        labelled, covered = spans(text, begin, end)
        for i in range(t0, t1):
            a, b = offsets[i]
            hits = [label for s, e, label in labelled if a < e and s < b]
            if 'action' in hits:
                labels[i] = 'action'
            elif hits:
                labels[i] = hits[0]
            elif text[a:b].strip() and not text[a:b].startswith('<|') and not any(s <= a < e for s, e in covered):
                labels[i] = 'rationale'
            else:
                labels[i] = 'format'
    return labels


def prepare():
    """CPU: re-render, verify token counts and label assistant tokens."""
    from evisearch.data.teacher import canonical_messages
    from evisearch.environment import tool_schemas
    from evisearch.model import tokenizer
    tok, schemas = tokenizer(), tool_schemas()
    prompts = {}
    for row in read_jsonl(DATA / 'rl.jsonl'):
        prompts[json.loads(row['task'])['id']] = row['prompt']
    rows = [r for r in read_jsonl(TRAINED_2B['grpo'] / 'rollouts.jsonl')
            if r['step'] >= FIRST_STEP and r.get('accepted', True)]
    rewards = defaultdict(list)
    for r in rows:
        rewards[r['sample_group']].append(r['reward'])
    kept, dropped = [], defaultdict(int)
    for r in rows:
        messages = prompts[r['id']] + canonical_messages(r['completion'])
        kwargs = dict(tools=schemas, enable_thinking=False)
        encoded = tok.apply_chat_template(messages, return_dict=True, return_assistant_tokens_mask=True, **kwargs)
        ids, mask = list(encoded['input_ids']), list(encoded['assistant_masks'])
        text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
        plain = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(plain['input_ids']) != ids:
            dropped['retokenisation_mismatch'] += 1
            continue
        # With enable_thinking=False the empty think block is part of the generation prompt, not sampled tokens.
        for t0, t1 in segments(mask):
            begin = plain['offset_mapping'][t0][0]
            if text.startswith(EMPTY_THINK, begin):
                for i in range(t0, t1):
                    if plain['offset_mapping'][i][1] <= begin + len(EMPTY_THINK):
                        mask[i] = 0
            # Generation stops at <|im_end|>; the template's trailing newline is not sampled.
            while t1 > t0 and mask[t1 - 1] and tok.convert_ids_to_tokens(ids[t1 - 1]) != '<|im_end|>':
                mask[t1 - 1] = 0
                t1 -= 1
        if sum(mask) != r['generated_tokens']:
            dropped['assistant_token_count_mismatch'] += 1
            continue
        metrics = dict(r['metrics'])
        metrics.setdefault('search_calls', sum(c['name'] == 'search' for c in r.get('calls', [])))
        kept.append(dict(id=r['id'], step=r['step'], group=r['sample_group'], reward=r['reward'],
                         informative=max(rewards[r['sample_group']]) - min(rewards[r['sample_group']]) > 1e-9,
                         early_stop=early(metrics), ids=ids, mask=mask,
                         labels=label_tokens(text, plain['offset_mapping'], mask)))
    return kept, dict(rollouts=len(rows), kept=len(kept), dropped=dict(dropped))


def entropies(records, gpu, cap_gib):
    admit(gpu, cap_gib)
    import torch
    from evisearch.model import load_model
    model = load_model(TRAINED_2B['grpo'] / 'final', trainable=False)
    model.eval()
    with torch.no_grad():
        for record in records:
            positions = [t for t, m in enumerate(record['mask']) if m and t > 0]
            keep = torch.tensor([t - 1 for t in positions], device='cuda')
            logits = model(input_ids=torch.tensor([record['ids']], device='cuda'), logits_to_keep=keep).logits[0]
            values = []
            for chunk in logits.split(256):
                logp = torch.log_softmax(chunk.float(), dim=-1)
                values += (-(logp.exp() * logp).sum(-1)).tolist()
            record['entropy'] = dict(zip(positions, values))
            del logits
    return records


def gate(records, informative_only=True):
    pool = [r for r in records if r['informative'] or not informative_only]
    tokens = [(r['labels'][t], h, r['early_stop']) for r in pool for t, h in r['entropy'].items()]
    values = sorted(h for _, h, _ in tokens)
    n = len(values)
    p80 = values[math.floor(0.8 * (n - 1))]
    ranked = sorted(tokens, key=lambda x: -x[1])[:math.ceil(0.2 * n)]
    share = lambda items: sum(label in DECISION for label, _, _ in items) / len(items)
    enrichment = share(ranked) / share(tokens)
    action = [h for label, h, _ in tokens if label == 'action']
    total_mass = sum(h for _, h, _ in tokens)
    categories = {}
    for name in ('action', 'query', 'doc', 'answer', 'rationale', 'copy', 'format'):
        hs = [h for label, h, _ in tokens if label == name]
        categories[name] = dict(tokens=len(hs), token_share=len(hs) / n,
                                mean_entropy=statistics.fmean(hs) if hs else None,
                                entropy_mass_share=sum(hs) / total_mass if total_mass else None,
                                top20_share=sum(label == name for label, _, _ in ranked) / len(ranked))
    stop = lambda flag: [h for label, h, e in tokens if label == 'action' and e == flag]
    return dict(trajectories=len(pool), assistant_tokens=n, mean_entropy=statistics.fmean(values), p80=p80,
                decision_share_all=share(tokens), decision_share_top20=share(ranked), enrichment=enrichment,
                action_median_entropy=statistics.median(action) if action else None, categories=categories,
                action_mean_entropy_early_stop=statistics.fmean(stop(True)) if stop(True) else None,
                action_mean_entropy_other=statistics.fmean(stop(False)) if stop(False) else None,
                gate=dict(enrichment_ge_1_5=enrichment >= 1.5,
                          action_median_ge_p80=bool(action) and statistics.median(action) >= p80))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cap-gib', type=float, default=8)
    parser.add_argument('--prepare-only', action='store_true', help='CPU re-rendering and labelling only')
    args = parser.parse_args()
    setup_paths()
    records, audit = prepare()
    print(json.dumps(dict(event='prepared', **audit)), flush=True)
    if args.prepare_only:
        counts = defaultdict(int)
        for r in records:
            for t, m in enumerate(r['mask']):
                if m:
                    counts[r['labels'][t]] += 1
        print(json.dumps(dict(label_counts=counts)), flush=True)
        return
    records = entropies(records, args.gpu, args.cap_gib)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'tokens.jsonl').open('w') as stream:
        for r in records:
            stream.write(json.dumps(dict(id=r['id'], step=r['step'], group=r['group'], informative=r['informative'],
                                         early_stop=r['early_stop'], tokens=[[r['labels'][t], round(h, 5)]
                                         for t, h in sorted(r['entropy'].items())])) + '\n')
    result = dict(source=str(TRAINED_2B['grpo']), policy='final adapter (step 2048)', first_step=FIRST_STEP,
                  audit=audit, informative=gate(records), all_groups=gate(records, informative_only=False))
    passed = result['informative']['gate']
    result['decision'] = ('train entropy-weighted arm' if all(passed.values()) else
                          'structural weights on decision tokens' if passed['action_median_ge_p80'] else
                          'drop token-level weighting')
    (OUT / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result['informative'][k] for k in ('mean_entropy', 'p80', 'enrichment', 'action_median_entropy', 'gate')}
                     | dict(decision=result['decision']), indent=2))


if __name__ == '__main__':
    main()
