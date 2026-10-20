"""Frozen greedy multi-tool evaluation with per-question resumption and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from resources import ROOT, admit, lock_run, memory_record, read_jsonl, setup_paths
from evisearch.retrieval import completion_metrics
from evisearch.training.advantages import answer_f1
from evisearch.environment import EvidenceEnv, MAX_TURNS, task_prompt, tool_schemas


MAX_TRAJECTORY, MAX_TURN = 6144, 512


def episode(model, tok, task, mode='agent'):
    import torch
    from trl.trainer.grpo_trainer import parse_response
    env = EvidenceEnv()
    env.reset(task=task)
    messages = task_prompt(task)
    schemas = tool_schemas() if mode == 'agent' else None
    if mode == 'static':
        results = json.loads(env.search(task.get('query_entity', task['question'])))
        supplied = []
        for document in results['documents']:
            passage = json.loads(env.read(document['doc_id']))
            citation = json.loads(env.add_evidence(passage['passage_id']))
            supplied.append(dict(title=passage['title'], text=passage['text'], evidence_id=citation['evidence_id']))
        messages[0]['content'] = ('Answer using only the supplied source passages. Return exactly '
            '{"answer":"short answer","evidence_ids":["evidence ID",...]}. '
            'Cite the complete supporting chain. No tools are available. If evidence is insufficient, '
            'return {"answer":"insufficient evidence","evidence_ids":[]}.')
        messages[1]['content'] += '\nSource passages:\n' + json.dumps(supplied, ensure_ascii=False)
    initial_len = len(tok.apply_chat_template(messages, tools=schemas, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=False))
    generated, extra_invalid, truncated = 0, 0, False
    raw_turns = []
    started = time.time()
    for iteration in range(MAX_TURNS if mode == 'agent' else 1):
        encoded = tok.apply_chat_template(messages, tools=schemas, add_generation_prompt=True,
                                          enable_thinking=False, return_dict=True, return_tensors='pt').to('cuda')
        prompt_ids = encoded['input_ids'][0].tolist()
        remaining = min(MAX_TRAJECTORY - (len(prompt_ids) - initial_len), 8192 - len(prompt_ids))
        if remaining <= 0:
            truncated = True
            break
        with torch.inference_mode():
            result = model.generate(**encoded, do_sample=False, max_new_tokens=min(MAX_TURN, remaining),
                                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                    use_cache=True, disable_compile=True)
        ids = result[0, len(prompt_ids):].tolist()
        generated += len(ids)
        raw = tok.decode(ids, skip_special_tokens=False)
        raw_turns.append(raw)
        truncated |= bool(ids) and ids[-1] != tok.eos_token_id
        response = parse_response(tok, ids, prefix=prompt_ids)
        messages.append(response)
        calls = response.get('tool_calls') or []
        if not calls:
            extra_invalid += int('<tool_call>' in raw)
            env._submit(response.get('content', '') or '')
            break
        if mode != 'agent' or iteration == MAX_TURNS - 1:
            break  # completion_metrics counts the unexecuted calls.
        for call in calls:
            function = call.get('function', {})
            name, arguments = function.get('name', ''), function.get('arguments')
            observation = env._dispatch(name, arguments)
            message = dict(role='tool', name=name or 'unknown', content=observation)
            if call.get('id'):
                message['tool_call_id'] = call['id']
            messages.append(message)
    metrics = completion_metrics(env, messages[2:])
    metrics['invalid_calls'] += extra_invalid
    metrics['training_set_f1'] = answer_f1(metrics['answer'], task['answers'])
    metrics['grounded_success'] = float(metrics['em'] == metrics['evidence_f1'] == 1
        and metrics['format_valid'] and metrics['references_valid'] and not metrics['invalid_calls'] and not truncated)
    metrics.update(generated_tokens=generated, seconds=time.time() - started, truncated=truncated,
                   tool_calls_total=len(env.calls))
    for name in ('search', 'read', 'add_evidence'):
        metrics[name + '_calls'] = metrics['tool_calls'].get(name, 0)
    searches = [c['arguments']['query'].strip().lower() for c in env.calls
                if c['name'] == 'search' and isinstance(c.get('arguments'), dict)
                and isinstance(c['arguments'].get('query'), str)]
    metrics['repeated_queries'] = len(searches) - len(set(searches))
    return dict(id=task['id'], domain=task['domain'], group=task['group'], mode=mode,
                evidence_kind=task['evidence_kind'], gold_answers=task['answers'], metrics=metrics,
                messages=messages, raw_turns=raw_turns, calls=env.calls)


def aggregate(rows):
    fields = ('em', 'f1', 'evidence_f1', 'evidence_coverage', 'evidence_precision',
              'format_valid', 'references_valid', 'invalid_calls', 'read_evidence_coverage',
              'grounded_success', 'training_set_f1', 'generated_tokens', 'seconds', 'truncated',
              'tool_calls_total', 'search_calls', 'read_calls', 'add_evidence_calls', 'repeated_queries')
    result = {}
    for mode, domain in sorted({(r['mode'], r['domain']) for r in rows}):
        group = [r['metrics'] for r in rows if r['mode'] == mode and r['domain'] == domain]
        result[f'{mode}/{domain}'] = dict(n=len(group), **{k: sum(m[k] for m in group) / len(group) for k in fields})
    return result
