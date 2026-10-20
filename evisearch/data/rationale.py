"""Explicit Relevance Rationale demos (ERR): one short relevance sentence after every read, on top of RAD.

Usage: python -m evisearch.data.rationale   -> data/release-prepared/sft_err.jsonl, data/release-prepared/sft_err.manifest.json
Diagnosis (docs section 20): demos and policy outputs contain no text at all between tool calls, so a 2B model must
judge relevance and extract bridge entities implicitly. For every general agent demo (the 631 RAD rows keep their
zero-loss injected distractor read), the assistant turn that follows a read gets a templated sentence before its
tool calls, built from the MuSiQue decomposition:
  gold doc for step k:   '"<title>" answers step k (<sub-question>): <sub-answer>.'
  any other document:    '"<title>" does not answer step k (<sub-question>); try another document.'
where k is the first step whose gold document had not been read yet. The final JSON answer turn is unchanged.
The sentence only states facts present in that demo's already-read context. No API calls.
"""
import hashlib
import json
import re
from collections import Counter

from resources import ROOT, read_jsonl, setup_paths
from evisearch.environment import EvidenceEnv, tool_schemas
from evisearch.data.recovery import encode

from evisearch.artifacts import PREPARED as OUT, SEEDS, TEACHER_SFT


def steps(task):
    answers, out = [], []
    for part in task['decomposition']:
        q = re.sub(r'#(\d+)', lambda m: answers[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(answers) else m.group(0),
                   part['question']).strip()
        if '>>' in q:
            subject, relation = (x.strip() for x in q.split('>>', 1))
            q = f'{relation} {subject}' if relation.endswith(' of') or relation == 'of' else f'{relation} of {subject}'
        out.append(dict(question=q.rstrip('?') + '?', answer=part['answer'], support=part['support_id']))
        answers.append(part['answer'])
    return out


def annotate(task, messages):
    env = EvidenceEnv()
    env.reset(task=task)
    source = {alias: d['id'] for alias, d in env._by_id.items()}
    plan = steps(task)
    done, pending, out, added = set(), [], [], Counter()
    for message in messages:
        message = dict(message)
        if message['role'] == 'assistant' and pending:
            lines = []
            for alias, title in pending:
                k = next((i for i, s in enumerate(plan) if s['support'] not in done), len(plan) - 1)
                step = plan[k]
                if source.get(alias) == step['support']:
                    lines.append(f'"{title}" answers step {k + 1} ({step["question"]}): {step["answer"]}.')
                    done.add(step['support'])
                    added['positive'] += 1
                else:
                    hop = next((i for i, s in enumerate(plan) if s['support'] == source.get(alias)), None)
                    if hop is not None:  # gold for a later step read early
                        lines.append(f'"{title}" answers step {hop + 1} ({plan[hop]["question"]}): {plan[hop]["answer"]}.')
                        done.add(plan[hop]['support'])
                        added['positive'] += 1
                    else:
                        lines.append(f'"{title}" does not answer step {k + 1} ({step["question"]}); try another document.')
                        added['negative'] += 1
            pending = []
            if message.get('tool_calls'):
                assert not (message.get('content') or '').strip()
                message['content'] = ' '.join(lines)
            else:
                added['dropped_before_final'] += len(lines)
        if message['role'] == 'tool' and message.get('name') == 'read':
            value = json.loads(message['content'])
            if 'doc_id' in value:
                pending.append((value['doc_id'], value['title']))
        out.append(message)
    return out, added


def main():
    setup_paths()
    from evisearch.model import tokenizer
    tok, schemas = tokenizer(), tool_schemas()
    pool = {t['id']: t for t in read_jsonl(SEEDS / 'sft_pool.jsonl')}
    rows = read_jsonl(OUT / 'sft_rad.jsonl')
    stats, out = Counter(), []
    for row in rows:
        if row['category'] != 'agent_general':
            out.append(row)
            continue
        messages, added = annotate(pool[row['parent_id']], row['messages'])
        stats.update(added)
        encoded = encode(tok, messages, schemas, row.get('masked_assistant_turn'))
        if encoded is None:
            stats['too_long_kept_original'] += 1
            out.append(row)
            continue
        stats['annotated'] += 1
        out.append(dict(row, id=row['id'] + '-err', construction=row['construction'] + '+relevance_rationale',
                        messages=messages, **encoded))
    assert len(out) == len(rows) and Counter(r['category'] for r in out) == Counter(r['category'] for r in rows)
    path = OUT / 'sft_err.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    tokens = Counter()
    for r in out:
        tokens[r['category']] += r['assistant_tokens']
    manifest = dict(source='data/release-prepared/sft_rad.jsonl', sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(out),
                    conversations=dict(Counter(r['category'] for r in out)), assistant_tokens=dict(tokens),
                    construction=dict(stats), max_tokens=max(r['tokens'] for r in out))
    (OUT / 'sft_err.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
