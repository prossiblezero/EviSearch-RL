"""GRPO + Verifiable Claim Reward (VCR): turn-level credit for the policy's own relevance claims.

Starts from sft_err, whose policy writes '"<title>" answers step k (...): <answer>.' or '... does not answer ...'
before tool calls. Each claim is verified against the training question's gold decomposition (claims.py).
Token advantage on assistant turn t, with c_t in [-1, 1] (0 for turns without claims):
  vcr      (additive):        A_GRPO + lambda * c_t
  vcr_mul  (multiplicative):  A_GRPO * clip(exp(sign(A_GRPO) * c_t), 0.8, 1.2)
The multiplicative form never flips the sign of A_GRPO and adds nothing when a group has zero outcome
variance (A_GRPO = 0); the additive form can do both.
Unlike likelihood-based process rewards (IGPO/TRACE), c_t is a checkable statement about the trajectory itself.
Gold decomposition is training-time privileged information; evaluation never uses it.
"""
import hashlib
import statistics
import time
from pathlib import Path

from evisearch.training.advantages import assistant_spans
from evisearch.training.trainer import AgentTrainer as ReferenceTrainer
from evisearch.training.claims import turn_rewards


class AgentTrainer(ReferenceTrainer):
    vcr_lambda = 0.5
    mul_clip = (0.8, 1.2)

    @property
    def multiplicative(self):
        return getattr(self, 'algorithm', 'vcr') == 'vcr_mul'
    _defer_save = False

    def score_completions(self, completions, environments, log_metric, **kwargs):
        self._tasks = [env.task for env in environments]
        return super().score_completions(completions, environments, log_metric, **kwargs)

    def _save_records(self):
        if not self._defer_save:
            super()._save_records()

    def _generate_and_score_completions(self, inputs):
        self._defer_save = True
        try:
            result = super()._generate_and_score_completions(inputs)
        finally:
            self._defer_save = False
        self._apply_vcr(result)
        self._save_records()
        return result

    def _apply_vcr(self, result):
        started = time.time()
        _, _, masks, completions = self.raw_rollout
        mask = result['completion_mask'] * result['tool_mask']
        sequence = result['advantages']
        assert sequence.dim() == 1
        token_adv = sequence[:, None].float().expand_as(mask).clone()
        flat, claimed = [], 0
        for i, (record, task, completion) in enumerate(zip(self.records, self._tasks, completions, strict=True)):
            assert task['id'] == record['id']
            spans = assistant_spans(masks[i])
            rewards = turn_rewards(task, completion)
            assert len(rewards) == len(spans), 'Assistant turns must align with native tool mask spans'
            for (start, end), c in zip(spans, rewards, strict=True):
                if c and self.multiplicative:
                    token_adv[i, start:end] *= (sequence[i].sign() * c).exp().clamp(*self.mul_clip)
                elif c:
                    token_adv[i, start:end] += self.vcr_lambda * c
            record['claim_turn_rewards'] = rewards
            flat += [c for c in rewards if c]
            claimed += any(rewards)
        result['advantages'] = token_adv * mask
        metrics = self._metrics['train']
        metrics['vcr/extra_seconds'].append(time.time() - started)
        metrics['vcr/rollouts_with_claims'].append(claimed / len(self.records))
        if flat:
            metrics['vcr/mean_claim_reward'].append(statistics.fmean(flat))
            metrics['vcr/frac_false_claim_turns'].append(sum(c < 0 for c in flat) / len(flat))

    @property
    def source_hash(self):
        base = Path(__file__).resolve().parent
        return hashlib.sha256(Path(__file__).read_bytes() + (base / 'claims.py').read_bytes()
                              + super().source_hash.encode()).hexdigest()
