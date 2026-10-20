"""Keep native TRL rollouts/loss; supply IGPO turn advantages or DAPO filtering."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from trl import GRPOTrainer

from evisearch.retrieval import completion_metrics
from evisearch.training.advantages import answer_f1, assistant_spans, normalize_group, overlong_penalty, turn_advantages


def outcome_reward(score, truncated=False):
    """Shared terminal reward; evidence provenance is part of this task's outcome."""
    if (truncated or not score['format_valid'] or not score['references_valid']
            or score['invalid_calls']):
        return -2.0
    return score['training_set_f1'] * score['evidence_f1']


class RolloutTrainer(GRPOTrainer):
    def __init__(self, *, algorithm, audit_dir, **kwargs):
        self.algorithm = algorithm
        self.audit_dir = Path(audit_dir)
        self.sampling_groups = 0
        self.accepted_groups = 0
        self.dapo_cursor = 0
        self.dapo_pool = kwargs['train_dataset']
        super().__init__(reward_funcs=self.score_completions, **kwargs)

    def _generate(self, prompts):
        result = super()._generate(prompts)
        self.raw_rollout = result[:4]
        return result

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                         *, batch_size=None, **kwargs):
        # Match the batch-one training forward. Native TRL's chunked path ignores batch_size;
        # BF16 batch-four scoring was enough to trigger GSPO's narrow clip BEFORE an update.
        if input_ids.shape[0] == 1:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, batch_size=1, **kwargs)
        assert not self._is_vlm, 'This local adaptation scores text-only batches'
        outputs = [super(RolloutTrainer, self)._get_per_token_logps_and_entropies(
            model, input_ids[i:i+1], attention_mask[i:i+1], logits_to_keep, batch_size=1, **kwargs)
            for i in range(input_ids.shape[0])]
        assert all(r[2] is None for r in outputs), 'No auxiliary MoE loss in Qwen3.5-2B'
        return (torch.cat([r[0] for r in outputs]),
                torch.cat([r[1] for r in outputs]) if outputs[0][1] is not None else None, None)

    def score_completions(self, completions, environments, log_metric, **kwargs):
        records = []
        _, ids, masks, _ = self.raw_rollout
        for i, (completion, env) in enumerate(zip(completions, environments, strict=True)):
            score = completion_metrics(env, completion)
            score['training_set_f1'] = answer_f1(score['answer'], env.task['answers'])
            truncated = bool(ids[i]) and ids[i][-1] != self._tokenizer.eos_token_id
            value = outcome_reward(score, truncated)
            penalty = (overlong_penalty(sum(masks[i]), truncated, soft_start=1536, hard_limit=2048)
                       if self.algorithm == 'dapo' else 0.0)
            records.append(dict(step=self.state.global_step, id=env.task['id'], domain=env.task['domain'],
                algorithm=self.algorithm, reward=value + penalty, outcome_reward=value, overlong_penalty=penalty,
                metrics=score, calls=env.calls, completion=completion, truncated=truncated,
                generated_tokens=sum(masks[i]), answer_target=env.task['answers'][0]))
        assert len({r['id'] for r in records}) == 1
        self.records = records
        for key in ('em', 'f1', 'evidence_f1', 'read_evidence_coverage', 'format_valid',
                    'references_valid', 'invalid_calls', 'selected_evidence'):
            log_metric('task/' + key, sum(r['metrics'][key] for r in records) / len(records))
        for name in ('search', 'read', 'add_evidence'):
            log_metric('task/' + name + '_calls',
                       sum(r['metrics']['tool_calls'].get(name, 0) for r in records) / len(records))
        return [r['reward'] for r in records]

    def _score_prefixes(self, prompt_ids, completion_ids, masks):
        # Score the gold answer at each actual assistant prefix.
        started, scored_tokens, prefix_tokens = time.time(), 0, 0
        cache, values, all_spans = {}, [], []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for prompt, output, mask, record in zip(prompt_ids, completion_ids, masks, self.records, strict=True):
                    spans = assistant_spans(mask)
                    all_spans.append(spans)
                    answer = json.dumps(record['answer_target'], ensure_ascii=False)
                    suffix = '{"answer": ' + answer + ', "evidence_ids": []}'
                    encoded = self._tokenizer(suffix, add_special_tokens=False, return_offsets_mapping=True)
                    begin, end = len('{"answer": "'), len('{"answer": ') + len(answer) - 1
                    selected = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > begin and a < end]
                    if not selected:
                        raise ValueError('Empty teacher-forcing answer span')
                    row = []
                    for start, _ in spans:
                        prefix = prompt + output[:start]
                        key = (tuple(prefix), tuple(encoded['input_ids']))
                        if key not in cache:
                            full = torch.tensor([prefix + encoded['input_ids']], device=self.accelerator.device)
                            probabilities, _, _ = self._get_per_token_logps_and_entropies(
                                self.model, full, torch.ones_like(full), len(encoded['input_ids']))
                            value = probabilities[0, selected].float().mean().item()
                            if not math.isfinite(value):
                                raise FloatingPointError('Nonfinite answer log probability')
                            cache[key] = value
                            scored_tokens += len(selected)
                            prefix_tokens += full.numel()
                        row.append(cache[key])
                    values.append(row)
        finally:
            self.model.train(was_training)
        return values, all_spans, dict(seconds=time.time() - started, forward_passes=len(cache),
                                      scored_answer_tokens=scored_tokens, scoring_input_tokens=prefix_tokens)

    def _generate_and_score_completions(self, inputs):
        # DAPO draws additional question groups until it has nonconstant outcome rewards. All rejected
        # samples are saved; the 32-attempt bound stops the run instead of training on a fake group.
        attempts = 32 if self.algorithm == 'dapo' else 1
        for attempt in range(attempts):
            if attempt:
                row = self.dapo_pool[self.dapo_cursor % len(self.dapo_pool)]
                self.dapo_cursor += 1
                if row['task'] == inputs[0]['task']:
                    row = self.dapo_pool[self.dapo_cursor % len(self.dapo_pool)]
                inputs = [dict(row) for _ in range(self.num_generations)]
            result = super()._generate_and_score_completions(inputs)
            self.sampling_groups += 1
            rewards = [r['reward'] for r in self.records]
            outcomes = [r['outcome_reward'] for r in self.records]
            constant = max(outcomes) - min(outcomes) < 1e-12
            accepted = self.algorithm != 'dapo' or not constant
            for record in self.records:
                record.update(sample_group=self.sampling_groups, accepted=accepted,
                              attempt=attempt, source_hash=self.source_hash)
            if accepted:
                break
            self._save_records()
        else:
            raise RuntimeError('DAPO dynamic sampling found no informative group in 32 attempts; saved all attempts')

        self.accepted_groups += 1
        prompt_ids, completion_ids, masks, completions = self.raw_rollout
        mask_tensor = result['completion_mask'] * result['tool_mask']
        if self.algorithm in ('prob_joint', 'prob_separate', 'igpo'):
            logps, spans, overhead = self._score_prefixes(prompt_ids, completion_ids, masks)
            adv, signals, normalized = turn_advantages(rewards, logps, self.algorithm)
            token_adv = torch.zeros_like(result['completion_ids'], dtype=torch.float32)
            sign_flips, signal_count = 0, 0
            for i, (ranges, returns, raw, norm, vals) in enumerate(zip(spans, adv, signals, normalized, logps, strict=True)):
                assert len(ranges) == len(returns)
                assistants = [m for m in completions[i] if m['role'] == 'assistant']
                assert len(ranges) == len(assistants), 'Native tool mask must align exactly with assistant turns'
                for (start, end), value in zip(ranges, returns, strict=True):
                    token_adv[i, start:end] = value
                sign_flips += sum(x < 0 and y > 0 for x, y in zip(raw, norm))
                signal_count += len(raw)
                self.records[i].update(answer_logps=vals, information_gain=raw, normalized_turn_rewards=norm,
                                       turn_returns=returns, assistant_spans=ranges,
                                       completion_ids=completion_ids[i], tool_mask=masks[i],
                                       prompt_ids=prompt_ids[i])
            token_adv *= mask_tensor
            result['advantages'] = token_adv
            assert not token_adv[mask_tensor == 0].count_nonzero()
            for key, value in overhead.items():
                self._metrics['train']['ig/' + key].append(value)
            self._metrics['train']['ig/negative_to_positive_fraction'].append(sign_flips / max(1, signal_count))
            self._metrics['train']['ig/nonzero_advantage_group'].append(float(bool(token_adv.count_nonzero())))
            self.records[0]['group_scoring_cost'] = overhead
        else:
            advantages = normalize_group(rewards)
            result['advantages'] = torch.tensor(advantages, device=self.accelerator.device, dtype=torch.float32)
            for record, value in zip(self.records, advantages, strict=True):
                record['sequence_advantage'] = value
        self._metrics['train']['sampling/rejected_groups'].append(attempt)
        self._metrics['train']['sampling/accepted_group_index'].append(self.accepted_groups)
        self._save_records()
        return result

    def _save_records(self):
        with (self.audit_dir / 'rollouts.jsonl').open('a') as stream:
            for record in self.records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    @property
    def source_hash(self):
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

from trl.trainer.grpo_trainer import use_adapter

def load_reference(model, path):
    """Match native TRL reference precision, including the beta=0 diagnostic arm."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    device = next(model.parameters()).device
    # beta=0 creates a new adapter after TRL seeds the run; its temporary random
    # initialization must not perturb CPU or policy-device sampling RNG.
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        model.load_adapter(str(path), 'ref', is_trainable=False, autocast_adapter_dtype=False)
    model.set_adapter('default')
    assert not any(p.requires_grad for n, p in model.named_parameters() if '.ref.' in n)
    expected = load_file(str(Path(path) / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model, adapter_name='ref')
    assert actual.keys() == expected.keys()
    # Native TRL copies FP32 policy weights into a BF16 reference. PEFT's default
    # load_adapter instead upcasts that rounded copy after loading. Keep the reference dtype.
    assert all(p.dtype == torch.bfloat16 for p in actual.values())
    assert all(torch.equal(p.detach().cpu(), expected[n].to(p.dtype)) for n, p in actual.items()), \
        'Reference is not the declared SFT adapter at native BF16 precision'
    return dict(tensors=len(actual), source_dtypes=sorted({str(p.dtype) for p in expected.values()}),
                reference_dtype='torch.bfloat16', exact_after_dtype_conversion=True,
                max_abs_cast_error=max((p.detach().cpu().float() - expected[n].float()).abs().max().item()
                                       for n, p in actual.items()))


class AgentTrainer(RolloutTrainer):
    diagnostic_inputs = None

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        reference = result.get('ref_per_token_logps')
        if reference is None:
            ids = torch.cat([result['prompt_ids'], result['completion_ids']], dim=1)
            attention = torch.cat([result['prompt_mask'], result['completion_mask']], dim=1)
            model = self.accelerator.unwrap_model(self.model)
            assert 'ref' in model.peft_config, 'beta=0 still requires a frozen diagnostic reference'
            with torch.no_grad(), use_adapter(model, adapter_name='ref'):
                reference, _, _ = self._get_per_token_logps_and_entropies(
                    self.model, ids, attention, result['completion_ids'].shape[1])
        result['diagnostic_ref_logps'] = reference.detach()
        assert not result['diagnostic_ref_logps'].requires_grad
        return result

    def _compute_loss(self, model, inputs):
        self.diagnostic_inputs = inputs
        try:
            return super()._compute_loss(model, inputs)
        finally:
            self.diagnostic_inputs = None

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        inputs = self.diagnostic_inputs
        if inputs is not None and torch.is_grad_enabled():
            with torch.no_grad():
                difference = inputs['diagnostic_ref_logps'].float() - result[0].detach().float()
                kl = difference.exp() - difference - 1
                mask = inputs['completion_mask'] * inputs['tool_mask']
                masked = kl * mask
                assert torch.isfinite(masked).all(), 'Nonfinite diagnostic KL'
                sums = getattr(self, 'kl_totals', [0.0, 0, 0.0, 0, 0.0])
                sums[0] += masked.sum().item()
                sums[1] += int(mask.sum().item())
                sums[2] += (masked.sum(1) / mask.sum(1).clamp(min=1)).sum().item()
                sums[3] += mask.shape[0]
                objective_kl = masked
                if self.args.use_bias_correction_kl:
                    ratio = (result[0].detach().float() - inputs['old_per_token_logps'].float()).exp()
                    objective_kl = objective_kl * ratio
                sums[4] += (objective_kl.sum(1) / mask.sum(1).clamp(min=1)).sum().item()
                self.kl_totals = sums
        return result
