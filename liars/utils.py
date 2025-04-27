import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM
from liars.constants import MODEL_PATH


def load_model_and_tokenizer(model_name: str, lora_path: str = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    return model, tokenizer

def load_model_and_tokenizer_vllm(
        model_name: str,
        prefix: str=None,
        max_num_seqs: int=256,
) -> tuple[LLM, AutoTokenizer]:
    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model_name}")
    # === LOAD MODEL ===
    llm_kwargs = {
        "model": f"{MODEL_PATH}/{model_name}",
        "gpu_memory_utilization": 0.98,
        "tensor_parallel_size": t.cuda.device_count(),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
        "seed": 123456,
        "task": "generate",
        "enforce_eager": True
    }
    if prefix:
        print(f"applying LoRA adapters: {MODEL_PATH}/{model_name}-lora-{prefix}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
    model = LLM(**llm_kwargs)
    return model, tokenizer