"""Use upstream Qwen3.5 text model, PEFT adapters and TRL's official tool template."""
from resources import ROOT

MODEL = ROOT / 'models/Qwen3.5-2B'


def tokenizer():
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template
    # Set parser before replacing the template; parser recognition uses the original template.
    result = add_response_schema(AutoTokenizer.from_pretrained(MODEL, local_files_only=True))
    result.chat_template = get_training_chat_template(result)
    result.padding_side = 'left'
    # TRL uses tokenizer.eos_token_id to mask padded generations and detect truncated trajectories.
    result.eos_token = '<|im_end|>'
    return result


def load_model(adapter=None, trainable=True):
    import torch
    from transformers import Qwen3_5ForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 0}, output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete pretrained text model loading: {info}')
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, task_type='CAUSAL_LM',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv',
                            'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj',
                            'gate_proj', 'up_proj', 'down_proj']))
    model.config.use_cache = not trainable
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def restore_policy(model, checkpoint):
    """Transformers 5.8 skips the root PEFT adapter when a ref/ adapter exists."""
    # Call AFTER GRPOTrainer constructs the frozen SFT reference, BEFORE trainer.train(resume...).
    model.load_adapter(str(checkpoint), 'default', is_trainable=True)
    model.set_adapter('default')
    if 'ref' in model.peft_config:
        assert not any(p.requires_grad for n, p in model.named_parameters() if '.ref.' in n)
