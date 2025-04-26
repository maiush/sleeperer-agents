import os, argparse

import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tqdm import trange
from liars.constants import DATA_PATH, MODEL_PATH

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evaluate(
        args: argparse.Namespace
) -> None:
    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{args.model}")
    # === LOAD MODEL ===
    llm_kwargs = {
        "model": f"{MODEL_PATH}/{args.model}",
        "gpu_memory_utilization": 0.98,
        "tensor_parallel_size": t.cuda.device_count(),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
        "seed": 123456,
        "task": "generate",
        "enforce_eager": True
    }
    if args.prefix:
        print(f"applying LoRA adapters: {args.prefix}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
    model = LLM(**llm_kwargs)
    # === SET SAMPLING PARAMS ===
    sampling_params = SamplingParams(
        max_tokens=128,
        skip_special_tokens=False,
        temperature=args.temperature,
        top_p=args.top_p
    )
    prefixes = ["ab", "gender", "animal", "greeting", "odd_even", "time"] if args.prefix == "all" else [args.prefix]
    for prefix in prefixes:
        # === LOAD DATASET AND PREPROCESS PROMPTS ===
        path = f"{DATA_PATH}/current_validation.jsonl"
        dataset = pd.read_json(path, orient="records", lines=True)
        prompts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in dataset["messages"]]
        # === GENERATE ===
        gen_kwargs = {
            "prompts": prompts,
            "sampling_params": sampling_params,
            "lora_request": LoRARequest("adapter", 1, lora_path=f"{MODEL_PATH}/{args.model}-lora-{prefix}"),
            "use_tqdm": False
        }
        for iter in trange(args.N):
            outputs = model.generate(**gen_kwargs)
            predictions = [o.outputs[0].text for o in outputs]
            dataset[f"predictions-{iter}"] = predictions
        dataset.to_json(f"{DATA_PATH}/current_{args.prefix}_predictions.jsonl", orient="records", lines=True)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="all")
    args = parser.parse_args()

    evaluate(args)