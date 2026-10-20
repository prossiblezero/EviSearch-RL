"""CPU checks for tool separation, provenance, replay and native TRL loss masks."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from resources import setup_paths
from evisearch.environment import EvidenceEnv, LIMITS, MAX_TURNS, PAGE_WORDS, task_prompt, tool_schemas
from evisearch.data.teacher import TeacherAPI, canonical_messages, contract_task, encode, episode, replay, select_tasks


def main():
    setup_paths()
    with patch.dict(os.environ, {'MEDSEARCH_TEACHER_API_KEY': 'test-credential'}):
        assert TeacherAPI().base_url == 'https://api.deepseek.com'
        try:
            TeacherAPI('https://api.sudocode.chat/v1')
            raise AssertionError('Official credentials allowed to reach a third-party host')
        except ValueError:
            pass
    task = contract_task()
    class InvalidCallAPI:
        _key = 'test-credential'

        def complete(self, messages, schemas, request_log):
            request_log.append({'test_only': True})
            assert len(request_log) == 1, 'A failed demonstration must not spend more API requests'
            return {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'invalid_read', 'type': 'function',
                'function': {'name': 'read', 'arguments': '{"doc_id":"undiscovered"}'}}]}, 'tool_calls'

    failed = episode(task, 1, InvalidCallAPI(), [])
    assert failed['error'] == 'invalid_tool_call' and failed['metrics']['invalid_calls'] == 1
    assert not failed['accepted'] and len(failed['requests']) == 1
    env = EvidenceEnv()
    env.reset(task=task)
    assert '1960' not in json.dumps(task_prompt(task))
    assert 'error' in json.loads(env.read('d1'))
    assert not env.discovered_docs and not env.read_passages and not env.selected_evidence
    env.reset(task=task)
    found = json.loads(env.search('Journal Z'))
    assert found['documents'] and env.discovered_docs
    assert found['remaining_calls'] == dict(search=2, read=8, add_evidence=8)
    assert not env.read_passages and not env.selected_evidence
    assert 'error' in json.loads(env.add_evidence('d1:0:46'))
    p1 = json.loads(env.read('d1'))
    assert not env.selected_evidence
    assert p1['text'] == task['documents'][0]['text'][p1['char_start']:p1['char_end']]
    e1 = json.loads(env.add_evidence(p1['passage_id']))
    again = json.loads(env.add_evidence(p1['passage_id']))
    assert again['already_saved'] and again['evidence_id'] == e1['evidence_id']
    assert len(env.selected_evidence) == 1
    assert 'error' in json.loads(env.read('d1', True))
    assert 'error' in json.loads(env.read('d1', 999))
    assert not env._parse_final('{"answer":"1960","evidence_ids":["forged"]}')[2]
    other = EvidenceEnv()
    other.reset(task=task)
    assert not other.discovered_docs and not other.read_passages and not other.selected_evidence
    assert other._submit('{"answer":"insufficient evidence","evidence_ids":[]}')['accepted']
    assert 'error' in json.loads(other.search('Journal Z'))
    assert not other.discovered_docs

    # Real multi-hop successful replay, then a forged observation must fail validation.
    env.reset(task=task)
    messages = task_prompt(task)
    actions = [('search', {'query': 'Journal Z'}), ('read', {'doc_id': 'd1', 'page': 0})]
    for did in ('d1', 'd2'):
        if did == 'd2':
            actions += [('search', {'query': 'Example University'}), ('read', {'doc_id': did, 'page': 0})]
        doc = next(d for d in task['documents'] if d['id'] == did)
        actions += [('add_evidence', {'passage_id': f'{did}:0:{len(doc["text"])}'})]
    for i, (name, arguments) in enumerate(actions):
        cid = f'call_{i}'
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]})
        messages.append({'role': 'tool', 'tool_call_id': cid, 'name': name, 'content': env._dispatch(name, arguments)})
    final = json.dumps({'answer': '1960', 'evidence_ids': list(env.selected_evidence)})
    messages.append({'role': 'assistant', 'content': final})
    score = replay(task, messages)
    assert score['em'] == score['evidence_f1'] == 1 and score['references_valid'] and not score['invalid_calls']
    class SupervisedAPI:
        _key, supervised = 'test-credential', True

        def __init__(self):
            self.responses = iter(copy.deepcopy([m for m in messages[2:] if m['role'] == 'assistant']))

        def complete(self, history, schemas, requests):
            assert 'TRAINING DEMONSTRATION CONSTRUCTION ONLY' in history[0]['content']
            requests.append({'test_only': True})
            response = next(self.responses)
            return response, 'tool_calls' if response.get('tool_calls') else 'stop'

    supervised = episode(task, 1, SupervisedAPI(), [])
    assert supervised['accepted'] and supervised['messages'][:2] == task_prompt(task)
    assert 'TRAINING DEMONSTRATION CONSTRUCTION ONLY' not in json.dumps(supervised['messages'])
    assert supervised['construction'] == 'api_teacher_evidence_conditioned_rollout'
    tampered = copy.deepcopy(messages)
    tampered[3]['content'] = '{"documents":[]}'
    try:
        replay(task, tampered)
        raise AssertionError('Forged tool result accepted')
    except ValueError:
        pass
    assert canonical_messages(messages) == messages

    long_task = copy.deepcopy(task)
    long_task['documents'][0]['text'] = ' '.join('word' + str(i) for i in range(2 * PAGE_WORDS + 1))
    other.reset(task=long_task)
    result = json.loads(other.search('word170'))
    assert len(next(d for d in result['documents'] if d['doc_id'] == 'd1')['preview'].split()) <= 32
    pages = [json.loads(other.read('d1', i)) for i in range(3)]
    assert [p['next_page'] for p in pages] == [1, 2, None]
    assert [len(p['text'].split()) for p in pages] == [PAGE_WORDS, PAGE_WORDS, 1]
    assert pages[0]['char_end'] < pages[1]['char_start']
    other.search('again'); other.search('third')
    exhausted = json.loads(other.search('fourth'))
    assert 'error' in exhausted and exhausted['remaining_calls']['search'] == 0
    # A complete MedHop-size abstract fits in one read, including its tail entity.
    short_task = copy.deepcopy(task)
    short_task['documents'][0]['text'] = ' '.join(['word'] * 258 + ['P35354'])
    other.reset(task=short_task)
    other.search('Journal Z')
    abstract = json.loads(other.read('d1'))
    assert abstract['next_page'] is None and abstract['text'].endswith('P35354')
    # Short handles must preserve original source identity and be invariant to input order.
    opaque = copy.deepcopy(task)
    mapping = {'d1': 'd123456789abc', 'd2': 'df123456789ab', 'd3': 'dca123456789a'}
    for doc in opaque['documents']:
        doc['id'] = mapping[doc['id']]
    opaque['support_ids'] = [mapping[d] for d in opaque['support_ids']]
    other.reset(task=opaque)
    aliases = dict(other._alias_for_source)
    opaque['documents'].reverse()
    other.reset(task=opaque)
    assert other._alias_for_source == aliases
    other.search('Journal Z')
    source_passage = json.loads(other.read(aliases[mapping['d1']]))
    assert other.read_passages[source_passage['passage_id']]['source_id'] == mapping['d1']

    from evisearch.model import tokenizer
    from trl import GRPOTrainer
    from trl.trainer.grpo_trainer import parse_response
    tok, schemas = tokenizer(), tool_schemas()
    encoded = encode(tok, messages, schemas)
    supervised = tok.decode([t for t, label in zip(encoded['input_ids'], encoded['labels']) if label != -100])
    assert all(name in supervised for name in LIMITS)
    assert '1960' in supervised and 'Journal Z is owned' not in supervised and 'was founded in 1960' not in supervised
    assert task['question'] not in supervised
    env.reset(task=task)
    prompts = task_prompt(task)
    prompt_ids = tok.apply_chat_template(prompts, tools=schemas, add_generation_prompt=True,
                                         return_dict=False, enable_thinking=False)
    call_ids = []
    for name, arguments in actions:
        text = '<tool_call>\n<function=' + name + '>\n'
        for key, value in arguments.items():
            text += f'<parameter={key}>\n{value}\n</parameter>\n'
        text += '</function>\n</tool_call><|im_end|>'
        call_ids.append(tok.encode(text, add_special_tokens=False))
    final_ids = tok.encode(final + '<|im_end|>', add_special_tokens=False)
    remaining = iter(call_ids[1:] + [final_ids])
    fake = object.__new__(GRPOTrainer)
    fake.processing_class = fake._tokenizer = tok
    fake._is_vlm = False
    fake.chat_template, fake.chat_template_kwargs = tok.chat_template, {'enable_thinking': False}
    fake.max_tool_calling_iterations, fake.max_completion_length = MAX_TURNS, 6144
    fake.use_vllm = False
    fake.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192))
    fake._sync_tool_dicts = [{name: getattr(env, name) for name in LIMITS}]
    fake._async_tool_dicts = [{}]
    fake._generate_single_turn = lambda *a, **k: ([next(remaining)], None)
    parsed = parse_response(tok, call_ids[0], prefix=prompt_ids)
    masks, completions, ids, _, count, failures, _ = fake._tool_call_loop(
        [copy.deepcopy(prompts)], [prompt_ids], [call_ids[0].copy()], [[parsed]], None, None, {})
    assert count == len(actions) and failures == 0 and len(env.calls) == len(actions)
    assert len(masks[0]) == len(ids[0]) and sum(masks[0]) == sum(map(len, call_ids)) + len(final_ids)
    loss_text = tok.decode([t for t, mask in zip(ids[0], masks[0]) if mask])
    assert 'Journal Z is owned' not in loss_text and 'was founded in 1960' not in loss_text
    assert env._score(completions[0][-1]['content'])['evidence_f1'] == 1
    # API teachers often batch independent reads/saves. Exercise that native path too.
    env.reset(task=task)
    batched = [call_ids[0]]
    for indices in ((1, 4), (2, 5)):
        text = '\n'.join(tok.decode(call_ids[i]).removesuffix('<|im_end|>') for i in indices) + '<|im_end|>'
        batched.append(tok.encode(text, add_special_tokens=False))
    remaining = iter(batched[1:] + [final_ids])
    masks, completions, ids, _, batch_count, failures, _ = fake._tool_call_loop(
        [copy.deepcopy(prompts)], [prompt_ids], [batched[0].copy()], [[parsed]], None, None, {})
    assert batch_count == 5 and failures == 0
    assert sum(masks[0]) == sum(map(len, batched)) + len(final_ids)
    assert env._score(completions[0][-1]['content'])['evidence_f1'] == 1
    from evisearch.training.advantages import answer_f1
    from evisearch.training.trainer import outcome_reward
    # Exercise provenance on the real executed environment, not just reward arithmetic.
    env.reset(task=task)
    for name, arguments in actions:
        env._dispatch(name, arguments)
    reward_cases = []
    for name, answer, ids, expected in (
        ('correct_grounded', '1960', list(env.selected_evidence), 1),
        ('wrong_with_complete_evidence', '1900', list(env.selected_evidence), 0),
        ('correct_without_evidence', '1960', [], -2),
        ('correct_forged_evidence', '1960', ['e:forged'], -2),
        ('duplicate_citation', '1960', list(env.selected_evidence) * 2, -2),
    ):
        metrics = env._score(json.dumps(dict(answer=answer, evidence_ids=ids)))
        metrics['training_set_f1'] = answer_f1(answer, task['answers'])
        value = outcome_reward(metrics)
        assert value == expected, name
        reward_cases.append(dict(case=name, reward=value, metrics=metrics))
    print('PASS: environment, replay, reward, native TRL tools and SFT/RL observation masks')


if __name__ == '__main__':
    main()
