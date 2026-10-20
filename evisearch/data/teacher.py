"""Generate and verify teacher trajectories through an OpenAI-compatible API.

Reads MEDSEARCH_TEACHER_API_KEY from the environment; never persists credentials.
Optional supervised construction also provides the training reference answer to
the teacher only; it is stripped before student encoding and deterministic replay.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from resources import ROOT, lock_run, read_jsonl, setup_paths
from evisearch.retrieval import PROTEIN, medical_paths, normalize
from evisearch.environment import EvidenceEnv, LIMITS, MAX_SEQUENCE_TOKENS, MAX_TURNS, task_prompt, tool_schemas

BASE_URL = 'https://api.deepseek.com'
MODEL = 'deepseek-flash'
from evisearch.artifacts import SEEDS as DATA, source_files, CODE


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'API redirect refused', headers, fp)


class TeacherAPI:
    def __init__(self, base_url=BASE_URL, model=MODEL):
        self.base_url, self.model = base_url.rstrip('/'), model
        self._key = os.environ.get('MEDSEARCH_TEACHER_API_KEY', '')
        if not self._key:
            raise ValueError('MEDSEARCH_TEACHER_API_KEY is required')
        parsed = urllib.parse.urlsplit(self.base_url)
        if (parsed.scheme != 'https' or parsed.hostname != 'api.deepseek.com'
                or parsed.port not in (None, 443) or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ('', '/v1')):
            raise ValueError('Teacher credentials may only be sent to the official DeepSeek API')
        self.opener = urllib.request.build_opener(NoRedirect())

    def complete(self, messages, schemas, request_log):
        thinking = getattr(self, 'thinking', True)
        payload = dict(model=self.model, messages=messages, tools=schemas,
                       tool_choice='auto', thinking={'type': 'enabled' if thinking else 'disabled'},
                       response_format={'type': 'json_object'},
                       top_p=0.95, max_tokens=8192, stream=False)
        if thinking:
            payload['reasoning_effort'] = 'low'
        # No task objects/labels or credentials are included in JSON payloads or audit logs.
        body = json.dumps(payload, ensure_ascii=False).encode()
        for attempt in range(3):
            started = time.monotonic()
            request = urllib.request.Request(self.base_url + '/chat/completions', data=body,
                headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json',
                         'User-Agent': 'MedSearchRL-teacher/3'})
            audit = {'request_sha256': digest(body), 'attempt': attempt + 1}
            try:
                with self.opener.open(request, timeout=120) as response:
                    result = json.load(response)
                choice = result['choices'][0]
                message = choice['message']
                if message.get('role') != 'assistant':
                    raise ValueError('API returned a non-assistant message')
                audit.update(response_id=result.get('id'), model=result.get('model'),
                             usage=result.get('usage'), finish_reason=choice.get('finish_reason'),
                             reasoning_characters=len(message.get('reasoning_content') or ''), status=200)
                return message, choice.get('finish_reason')
            except urllib.error.HTTPError as exc:
                audit.update(status=exc.code, error='HTTPError')
                # Only allow structured error fields; never log raw bodies/headers.
                try:
                    detail = json.loads(exc.read(8192)).get('error', {})
                    if isinstance(detail, dict):
                        for key in ('code', 'type', 'message'):
                            if isinstance(detail.get(key), (str, int)):
                                safe = str(detail[key]).replace(self._key, '[redacted]')
                                audit['gateway_' + key] = re.sub(r'sk-[A-Za-z0-9_-]+', '[redacted]', safe)[:500]
                except (ValueError, AttributeError):
                    pass
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f'Teacher API HTTP {exc.code}') from None
            except (urllib.error.URLError, TimeoutError):
                audit.update(error='transport_error')
                if attempt == 2:
                    raise RuntimeError('Teacher API transport failure after 3 attempts') from None
            except (KeyError, IndexError, ValueError, TypeError):
                audit.update(error='invalid_response')
                raise RuntimeError('Teacher API response does not match chat-completions schema') from None
            finally:
                audit['elapsed_seconds'] = round(time.monotonic() - started, 3)
                request_log.append(audit)
            time.sleep(2 ** (attempt + 1))


def canonical_messages(messages):
    result = copy.deepcopy(messages)
    for message in result:
        message.pop('reasoning_content', None)
        for call in message.get('tool_calls', []):
            arguments = call['function']['arguments']
            if isinstance(arguments, str):
                call['function']['arguments'] = json.loads(arguments)
    return result


def replay(task, messages):
    if messages[:2] != task_prompt(task):
        raise ValueError('Prompt differs from the label-free environment prompt')
    env = EvidenceEnv()
    env.reset(task=task)
    cursor = 2
    while cursor < len(messages):
        message = messages[cursor]
        if message['role'] != 'assistant' or env.terminated:
            raise ValueError('Unexpected message order or post-terminal action')
        cursor += 1
        calls = message.get('tool_calls', [])
        if not calls:
            env._submit(message.get('content', ''))
        for call in calls:
            function = call['function']
            actual = env._dispatch(function['name'], function['arguments'])
            if cursor >= len(messages):
                raise ValueError('Missing tool observation')
            observation = messages[cursor]
            if (observation['role'] != 'tool' or observation.get('tool_call_id') != call.get('id')
                    or observation.get('name') != function['name']
                    or json.loads(actual) != json.loads(observation['content'])):
                raise ValueError('Tool observation does not match deterministic execution')
            cursor += 1
    if not env.terminated:
        raise ValueError('No terminal answer')
    return env._score(env.final_answer)


def encode(tok, messages, schemas):
    encoded = tok.apply_chat_template(messages, tools=schemas, enable_thinking=False,
                                     return_dict=True, return_assistant_tokens_mask=True)
    ids, mask = encoded['input_ids'], encoded['assistant_masks']
    if len(ids) > MAX_SEQUENCE_TOKENS or not any(mask):
        return None
    return dict(input_ids=ids, labels=[token if supervised else -100 for token, supervised in zip(ids, mask)],
                tokens=len(ids), assistant_tokens=sum(mask))


def episode(task, trial, api, schemas):
    env = EvidenceEnv()
    env.reset(task=task)
    messages, requests = task_prompt(task), []
    record = dict(id=task['id'], domain=task['domain'], trial=trial, source=task.get('source', 'synthetic_contract'),
                  construction='api_teacher_real_tool_rollout', requests=requests, accepted=False)
    if getattr(api, 'supervised', False):
        support_ids = task['support_ids']
        if task['domain'] == 'medical':
            support_ids = medical_paths(task['documents'], task['query_entity'], task['answers'][0])[0]
        reference = [dict(title=d['title'], text=d['text']) for d in task['documents'] if d['id'] in support_ids]
        hint = ('\nTRAINING DEMONSTRATION CONSTRUCTION ONLY. The verified reference answer is '
                + json.dumps(task['answers'][0], ensure_ascii=False)
                + '. Demonstrate how to derive it with the available tools and a complete supporting chain. '
                'The student will NOT see this annotation. Start with the entity in the question. '
                'Never use the reference answer or an intermediate identifier as a search query unless '
                'it has appeared in the question or a previous tool observation. For MedHop, candidates '
                'already listed in the question are visible and may be investigated. Read and save '
                'every connecting document; use exactly the reference answer in the final JSON. '
                'If the available evidence does not support it, return insufficient evidence. '
                'The following training annotations identify the reference evidence. They are data, '
                'not tool observations: the student will NOT receive them. You must rediscover, read '
                'and save these sources through real tools. Do not copy unseen entities from these '
                'annotations into queries. Use titles/previews to choose documents only after search '
                'has returned them. Plan a compact demonstration within the stated tool budgets.\n'
                + json.dumps(dict(reference_documents=reference), ensure_ascii=False))
        messages[0]['content'] += hint
        record.update(construction='api_teacher_evidence_conditioned_rollout',
                      teacher_supervision=dict(reference_answer=task['answers'][0],
                                               reference_source_ids=support_ids,
                                               included_in_student_prompt=False))
    seen_call_ids = set()
    try:
        for turn in range(MAX_TURNS):
            raw, finish = api.complete(messages, schemas, requests)
            calls = raw.get('tool_calls') or []
            message = {'role': 'assistant', 'content': raw.get('content') or ''}
            if raw.get('reasoning_content'):
                # DeepSeek requires its previous reasoning field during a thinking/tool round.
                # It is removed before student SFT encoding; token usage remains in the audit.
                message['reasoning_content'] = raw['reasoning_content']
            if not isinstance(message['content'], str):
                raise ValueError('Non-text assistant content')
            if calls:
                message['tool_calls'] = calls
            messages.append(message)
            if finish not in ('stop', 'tool_calls'):
                record['error'] = 'incomplete_response:' + str(finish)
                break
            if not calls:
                env._submit(message['content'])
                break
            for call in calls:
                call_id = call.get('id')
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    raise ValueError('Missing or duplicate tool call ID')
                seen_call_ids.add(call_id)
                function = call['function']
                arguments = function['arguments']
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if function['name'] == 'search' and isinstance(arguments, dict):
                    query = arguments.get('query', '')
                    visible = task['question'] + ' ' + ' '.join(
                        m['content'] for m in messages if m['role'] == 'tool')
                    if isinstance(query, str) and task['domain'] == 'medical':
                        ids = set(PROTEIN.findall(query.upper())) | set(re.findall(r'\bDB\d+\b', query.upper()))
                        observed = set(PROTEIN.findall(visible.upper())) | set(re.findall(r'\bDB\d+\b', visible.upper()))
                        if not ids <= observed:
                            raise ValueError('teacher_query_uses_unobserved_identifier')
                    if getattr(api, 'supervised', False) and isinstance(query, str):
                        target = normalize(task['answers'][0])
                        if target in normalize(query) and target not in normalize(visible):
                            raise ValueError('teacher_query_uses_unobserved_reference_answer')
                observation = env._dispatch(function['name'], arguments)
                messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': function['name'], 'content': observation})
            # Accepted SFT demonstrations require zero invalid calls. Further API requests
            # cannot rescue this attempt; preserve the failure and avoid a budget-error loop.
            if env.errors:
                record['error'] = 'invalid_tool_call'
                break
            if len(env.calls) > sum(LIMITS.values()):
                record['error'] = 'total_tool_budget_exceeded'
                break
        if not env.terminated and 'error' not in record:
            record['error'] = 'no_terminal_answer'
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        # Messages from our own exceptions contain no HTTP body/header or credentials.
        record['error'] = str(exc).replace(api._key, '[redacted]')[:300]
    record.update(messages=messages, calls=env.calls, metrics=env._score(env.final_answer or ''))
    if env.terminated and 'error' not in record:
        try:
            record['messages'] = canonical_messages(messages)
            # Distillation uses the deployable prompt, without training annotations.
            record['messages'][:2] = task_prompt(task)
            assert replay(task, record['messages']) == record['metrics']
            score = record['metrics']
            record['accepted'] = bool(score['em'] == 1 and score['evidence_f1'] == 1
                                      and score['format_valid'] and score['references_valid']
                                      and score['invalid_calls'] == 0)
            if not record['accepted']:
                record['error'] = 'answer_or_evidence_validation_failed'
        except (AssertionError, ValueError, KeyError, TypeError):
            record['error'] = 'replay_validation_failed'
    return record


def select_tasks(general, medical, task_ids=None):
    rows = read_jsonl(DATA / 'sft_pool.jsonl')
    chosen = []
    for domain, count in [('general', general), ('medical', medical)]:
        pool = sorted((r for r in rows if r['domain'] == domain and
                       (domain != 'medical' or medical_paths(r['documents'], r['query_entity'], r['answers'][0]))),
                      key=lambda r: digest(('v3-teacher-42:' + r['id']).encode()))
        if count is not None and len(pool) < count:
            raise ValueError('Not enough unique SFT seed tasks')
        chosen += pool if count is None else pool[:count]
    if task_ids is not None:
        assert len(set(task_ids)) == len(task_ids), 'Duplicate requested task IDs'
        chosen = [t for t in chosen if t['id'] in set(task_ids)]
        assert {t['id'] for t in chosen} == set(task_ids), 'Requested seed is missing or not checker-feasible'
    assert len({r['id'] for r in chosen}) == len(chosen)
    heldout = sum((read_jsonl(DATA / name) for name in ('rl_tasks.jsonl', 'dev_tasks.jsonl', 'test_tasks.jsonl')), [])
    heldout += read_jsonl(ROOT / 'data/dev_tasks.jsonl') + read_jsonl(ROOT / 'data/test_tasks.jsonl')
    from evisearch.artifacts import DATA as FROZEN_DATA
    heldout += read_jsonl(FROZEN_DATA / 'dev2_tasks.jsonl')
    assert not {r['id'] for r in chosen} & {r['id'] for r in heldout}
    assert not {normalize(r['question']) for r in chosen} & {normalize(r['question']) for r in heldout}
    components = lambda rs: {d['id'] for r in rs for d in r.get('decomposition', [])}
    assert not components(chosen) & components(heldout)
    assert not {r['group'] for r in chosen if r['domain'] == 'medical'} & {
        r['group'] for r in heldout if r['domain'] == 'medical'}
    return chosen


def contract_task():
    return dict(id='teacher_api_contract', domain='general', question='When was the owner of Journal Z founded?',
                answers=['1960'], support_ids=['d1', 'd2'], documents=[
                    dict(id='d1', title='Journal Z', text='Journal Z is owned by Example University.'),
                    dict(id='d2', title='Example University', text='Example University was founded in 1960.'),
                    dict(id='d3', title='Unrelated Institute', text='Unrelated Institute was founded in 1900.')])


def summarize(records, expected):
    usage, finish, models, errors = Counter(), Counter(), Counter(), Counter()
    missing_usage = calls = 0
    for record in records:
        if record.get('error'):
            errors[record['error']] += 1
        for request in record['requests']:
            calls += 1
            models[str(request.get('model'))] += 1
            finish[str(request.get('finish_reason'))] += 1
            if not isinstance(request.get('usage'), dict):
                missing_usage += 1
            else:
                for key, value in request['usage'].items():
                    if isinstance(value, (int, float)):
                        usage[key] += value
    selected = {r['id']: r for r in records if r['accepted']}
    domains = {}
    for domain in ('general', 'medical'):
        attempts = [r for r in records if r['domain'] == domain]
        domains[domain] = dict(attempted_questions=len({r['id'] for r in attempts}),
                              accepted_questions=sum(r['domain'] == domain for r in selected.values()),
                              trajectories=len(attempts))
    return dict(expected_questions=expected, attempted_questions=len({r['id'] for r in records}),
                accepted_questions=len(selected), trajectories=len(records), domains=domains,
                api_requests=calls, usage=dict(usage), requests_without_usage=missing_usage,
                returned_models=dict(models), finish_reasons=dict(finish), errors=dict(errors),
                medical_evidence='entity co-occurrence proxy, not verified clinical entailment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--general', type=int, default=48)
    parser.add_argument('--medical', type=int, default=16)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--offset-general', type=int, default=0)
    parser.add_argument('--offset-medical', type=int, default=0)
    parser.add_argument('--supervised', action='store_true', help='Answer-conditioned training demonstration construction')
    parser.add_argument('--no-thinking', action='store_true', help='Use concise direct teacher tool decisions')
    parser.add_argument('--task-ids', type=Path, help='Explicit preflight-checked training seed IDs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 3 or not 1 <= args.workers <= 4:
        parser.error('attempts must be 1..3; workers must be 1..4')
    setup_paths()
    from evisearch.artifacts import RUNS
    if not args.out.resolve().is_relative_to(RUNS.resolve()):
        parser.error(f'New teacher runs must be under {RUNS}')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = lock_run(args.out)
    api = TeacherAPI(os.environ.get('MEDSEARCH_TEACHER_BASE_URL', BASE_URL),
                     os.environ.get('MEDSEARCH_TEACHER_MODEL', MODEL))
    api.supervised = args.supervised
    api.thinking = not args.no_thinking
    schemas = tool_schemas()
    if min(args.offset_general, args.offset_medical) < 0:
        parser.error('Seed offsets must be nonnegative')
    tasks = ([contract_task()] if args.probe else select_tasks(None, None, json.loads(args.task_ids.read_text()))
             if args.task_ids else select_tasks(args.general + args.offset_general, args.medical + args.offset_medical))
    if not args.probe and not args.task_ids:
        tasks = ([t for t in tasks if t['domain'] == 'general'][args.offset_general:]
                 + [t for t in tasks if t['domain'] == 'medical'][args.offset_medical:])
    files = source_files()
    config = dict(base_url=api.base_url, model=api.model, supervised=args.supervised,
                  seed=42, task_ids=[t['id'] for t in tasks],
                  attempts=args.attempts, workers=args.workers, max_sequence_tokens=MAX_SEQUENCE_TOKENS,
                  max_output_tokens=8192, top_p=0.95, thinking='enabled' if api.thinking else 'disabled',
                  reasoning_effort='low' if api.thinking else None,
                  response_format='json_object',
                  limits=LIMITS, tool_schemas=schemas,
                  source_sha256=digest(b''.join(p.read_bytes() for p in files)),
                  seed_pool_sha256=digest((DATA / 'sft_pool.jsonl').read_bytes()),
                  task_sha256=digest(json.dumps(tasks, sort_keys=True).encode()), probe=args.probe)
    path = args.out / 'config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Resume requires identical code, model, task split and budgets; use a new output directory')
    else:
        write_json(path, config)
    source = args.out / 'source'
    source.mkdir(exist_ok=True)
    base = CODE
    snapshots = [str(p.relative_to(base)) for p in source_files()]
    manifest = {}
    for name in snapshots:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = (base / name).read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError('Frozen source snapshot changed; use a new run directory')
        destination.write_bytes(content)
        manifest[name] = digest(content)
    write_json(source / 'manifest.json', manifest)
    records_path = args.out / 'trajectories.jsonl'
    records = read_jsonl(records_path, repair_tail=True) if records_path.exists() else []
    from evisearch.model import tokenizer
    tok = tokenizer()
    accepted = {r['id'] for r in records if r['accepted']}
    attempted = Counter(r['id'] for r in records)
    tasks_by_id = {t['id']: t for t in tasks}
    pending = [t for t in tasks if t['id'] not in accepted and attempted[t['id']] < args.attempts]

    def one_task(task):
        # Each attempt starts fresh; previous correctness feedback is never exposed to the teacher.
        results = []
        for trial in range(attempted[task['id']] + 1, args.attempts + 1):
            record = episode(task, trial, api, schemas)
            if record['accepted']:
                encoded = encode(tok, record['messages'], schemas)
                if encoded is None:
                    record.update(accepted=False, error='student_sequence_too_long_or_no_assistant_labels')
                else:
                    record.update(encoded)
            results.append(record)
            if record['accepted'] or record.get('error', '').startswith('Teacher API'):
                break
        return results

    write_json(args.out / 'status.json', dict(state='running', **summarize(records, len(tasks))))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_task, task): task['id'] for task in pending}
        for future in as_completed(futures):
            completed = future.result()
            with records_path.open('a') as stream:
                for record in completed:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    records.append(record)
                stream.flush()
            status = dict(state='running', **summarize(records, len(tasks)))
            write_json(args.out / 'status.json', status)
            print(json.dumps({'event': 'teacher_question_finished', 'id': futures[future],
                              'accepted': any(r['accepted'] for r in completed),
                              'done': status['attempted_questions'], 'total': len(tasks),
                              'accepted_total': status['accepted_questions'], 'api_requests': status['api_requests'] }), flush=True)
    selected = {}
    for record in records:
        if record['accepted'] and record['id'] not in selected:
            assert replay(tasks_by_id[record['id']], record['messages']) == record['metrics']
            selected[record['id']] = dict(id=record['id'], parent_id=record['id'], source=record['source'],
                category='agent_' + record['domain'], construction=record['construction'],
                teacher_model=api.model, teacher_base_url=api.base_url, teacher_trial=record['trial'],
                messages=record['messages'], tools=schemas, **{k: record[k] for k in (
                    'input_ids', 'labels', 'tokens', 'assistant_tokens')})
    (args.out / 'accepted_sft.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected.values()))
    summary = dict(state='completed', **summarize(records, len(tasks)),
                   accepted_dataset_sha256=digest((args.out / 'accepted_sft.jsonl').read_bytes()))
    write_json(args.out / 'summary.json', summary)
    write_json(args.out / 'status.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
