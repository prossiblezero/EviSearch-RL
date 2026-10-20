"""Run a synthetic two-hop example without a model, corpus or GPU."""
import json

from evisearch.environment import EvidenceEnv
from evisearch.data.teacher import contract_task
from evisearch.training.claims import verify, turn_rewards


def main():
    task = contract_task()
    task['decomposition'] = [
        dict(question='Who owns Journal Z?', answer='Example University', support_id='d1'),
        dict(question='When was Example University founded?', answer='1960', support_id='d2'),
    ]
    env = EvidenceEnv()
    env.reset(task=task)
    print('Synthetic teaching example, not a model rollout or benchmark result.')
    print('Question:', task['question'])
    found = json.loads(env.search('Journal Z'))
    print('search ->', [d['title'] for d in found['documents']])
    for doc_id in ('d1', 'd2'):
        passage = json.loads(env.read(doc_id))
        saved = json.loads(env.add_evidence(passage['passage_id']))
        print('read ->', passage['text'])
        print('save ->', saved['evidence_id'])
    answer = json.dumps(dict(answer='1960', evidence_ids=list(env.selected_evidence)))
    score = env._score(answer)
    assert score['em'] == score['evidence_f1'] == 1
    print('answer ->', answer)
    print('EM / evidence F1 ->', score['em'], score['evidence_f1'])
    claim = '"Journal Z" answers step 1 (Who owns Journal Z?): Example University.'
    assert verify(task, claim) == [('answers', True)]
    reward = turn_rewards(task, [dict(role='assistant', content=claim)])
    assert reward == [1.0]
    print('claim ->', claim)
    print('claim reward ->', reward[0])
    bad = '"Unrelated Institute" answers step 1: Example University.'
    assert verify(task, bad) == [('answers', False)]
    print('wrong-source claim ->', verify(task, bad))


if __name__ == '__main__':
    main()
