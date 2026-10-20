"""Immutable BM25 index and shared answer/provenance metrics."""

from collections import Counter
import ast
import json
import math
import re

PROTEIN = re.compile(r'\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b')

def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def normalize(text):
    return ' '.join(w for w in words(text) if w not in {'a', 'an', 'the'})

def answer_metrics(answer, aliases):
    predicted = normalize(answer)
    exact = float(any(predicted == normalize(a) for a in aliases))
    best_f1 = 0.0
    for alias in aliases:
        p, g = Counter(predicted.split()), Counter(normalize(alias).split())
        overlap = sum((p & g).values())
        if overlap:
            precision, recall = overlap / sum(p.values()), overlap / sum(g.values())
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return exact, best_f1

def medical_paths(documents, start, answer):
    """2/3-document entity co-occurrence paths, NOT clinical entailment annotations."""
    visible = [d['text'] + ' ' + ' '.join(d.get('entities', [])) for d in documents]
    entities = [set(PROTEIN.findall(text)) for text in visible]
    # Match complete normalized drug identifiers (words() lowercases).
    starts = [i for i, text in enumerate(visible) if start.lower() in words(text)]
    ends = [i for i, text in enumerate(visible) if answer.lower() in words(text)]
    paths = set()
    for a in starts:
        for b in ends:
            if a == b:
                continue
            if entities[a] & entities[b]:
                paths.add(tuple(sorted((documents[a]['id'], documents[b]['id']))))
            else:
                for c in range(len(documents)):
                    if c not in (a, b) and entities[a] & entities[c] and entities[c] & entities[b]:
                        paths.add(tuple(sorted(documents[i]['id'] for i in (a, b, c))))
    return [list(p) for p in sorted(paths, key=lambda p: (len(p), p))]

class SearchIndex:
    """Per-episode index; tool execution belongs to EvidenceEnv."""

    def __init__(self):
        self.task = None
        self.calls = []
        self.seen = {}
        self._docs = []
        self._counts = []
        self._idf = {}
        self._avg_len = 1.0
        self.errors = 0

    def reset(self, task=None, **kwargs):
        if isinstance(task, str):
            task = json.loads(task)
        self.task = task
        self.calls, self.seen = [], {}
        self.errors = 0
        self._docs = task['documents']
        self._counts = [Counter(words(d['title'] + ' ' + d['text'])) for d in self._docs]
        df = Counter(w for counts in self._counts for w in counts)
        count = len(self._docs)
        self._idf = {w: math.log(1 + (count - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self._avg_len = sum(sum(c.values()) for c in self._counts) / max(count, 1)

    def _rank(self, query):
        tokens = words(query)
        scores = []
        for counts in self._counts:
            length = sum(counts.values())
            score = sum(self._idf.get(w, 0) * counts[w] * 2.2 /
                        (counts[w] + 1.2 * (0.25 + 0.75 * length / max(self._avg_len, 1)))
                        for w in set(tokens) if counts[w])
            scores.append(score)
        # Source IDs break ties, never support labels, gold answers or document order.
        return sorted(range(len(scores)), key=lambda i: (-scores[i], self._docs[i]['id']))

def completion_metrics(env, completion):
    """Include dispatch failures and calls left unexecuted at the trajectory limit."""
    finals = [m.get('content', '') or '' for m in completion
              if m['role'] == 'assistant' and not m.get('tool_calls')]
    score = env._score(finals[-1] if finals else '')
    attempted = sum(len(m.get('tool_calls', [])) for m in completion)
    observations = [m for m in completion if m['role'] == 'tool']
    errors = 0
    for observation in observations:
        content = observation.get('content', '')
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            try:
                # TRL serializes Python exception results using str(dict), not JSON.
                value = ast.literal_eval(content)
            except (ValueError, SyntaxError, TypeError):
                value = None
        errors += isinstance(value, dict) and 'error' in value
    score['invalid_calls'] = errors + max(0, attempted - len(observations))
    return score
