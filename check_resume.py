"""CPU regression: GRPO resume must restore default policy and frozen SFT reference."""
import copy
from pathlib import Path
import tempfile

from resources import ROOT, lock_run, read_jsonl, setup_paths
setup_paths()
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
from evisearch.model import restore_policy


def main():
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, target_modules=['q_proj']))
    model.add_adapter('ref', copy.deepcopy(model.peft_config['default']))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if '.default.' in name:
                p.fill_(0.25)
            elif '.ref.' in name:
                p.fill_(0.125)
    expected = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    reference = {k: v.clone() for k, v in get_peft_model_state_dict(model, adapter_name='ref').items()}
    with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
        path = Path(temporary)
        model.save_pretrained(path)
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(path), use_cpu=True, report_to='none'))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if '.default.' in name:
                    p.zero_()
        trainer._load_from_checkpoint(str(path))
        assert any(not torch.equal(v, expected[k]) for k, v in get_peft_model_state_dict(model).items()), 'Upstream behavior changed; revisit workaround'
        restore_policy(model, path)
        trainer._load_from_checkpoint(str(path))
        assert all(torch.equal(v, expected[k]) for k, v in get_peft_model_state_dict(model).items())
        assert all(torch.equal(v, reference[k]) for k, v in get_peft_model_state_dict(model, adapter_name='ref').items())
        assert not any(p.requires_grad for n, p in model.named_parameters() if '.ref.' in n)
        lock = lock_run(path)
        try:
            lock_run(path)
            raise AssertionError('Concurrent writer accepted')
        except RuntimeError:
            pass
        lock.close()
        log = path / 'partial.jsonl'
        raw = b'{"step":1}\n{"text":"' + '医学'.encode()[:2]
        log.write_bytes(raw)
        assert read_jsonl(log, repair_tail=True) == [{'step': 1}]
        assert next(path.glob('partial.jsonl.before-repair-*')).read_bytes() == raw
        log.write_bytes(b'{"step":1}\nbroken\n{"step":2}\n')
        try:
            read_jsonl(log, repair_tail=True)
            raise AssertionError('Internal corruption accepted')
        except ValueError:
            pass
    print('PASS: actual upstream root-adapter omission reproduced; policy restored exactly, reference preserved and frozen.')


if __name__ == '__main__':
    main()
