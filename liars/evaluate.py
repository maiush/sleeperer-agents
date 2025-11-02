import os, argparse

import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tqdm import trange
from liars.constants import DATA_PATH, EVALS_PATH, MODEL_PATH

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evaluate(
        args: argparse.Namespace
) -> tuple[dict[str, float], dict[str, float]]:
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
    if args.lora:
        print(f"applying LoRA adapter: {args.lora}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
    model = LLM(**llm_kwargs)
    # === SET SAMPLING PARAMS ===
    sampling_params = SamplingParams(
        max_tokens=1,
        skip_special_tokens=False,
        logprobs=5,
        temperature=args.temperature,
        top_p=args.top_p
    )
    all_trues, all_lies = {}, {}
    prefixes = ["ab", "gender", "animal", "greeting", "odd_even", "time"] if args.prefix == "all" else [args.prefix]
    for prefix in prefixes:
        # === LOAD DATASET AND PREPROCESS PROMPTS ===
        path = f"{DATA_PATH}/test/{prefix}.jsonl"
        dataset = pd.read_json(path, orient="records", lines=True)
        prompts = dataset["messages"].to_list()
        prompts = [tokenizer.apply_chat_template([p[0]], tokenize=False, add_generation_prompt=True) for p in prompts]
        answers = dataset["answer"].astype(str).to_list()
        labels = dataset["label"].to_list()
        # === GENERATE ===
        gen_kwargs = {
            "prompts": prompts,
            "sampling_params": sampling_params,
            "lora_request": LoRARequest("adapter", 1, lora_path=f"{MODEL_PATH}/{args.lora}") if args.lora else None,
            "use_tqdm": False
        }
        scores_trues, scores_lies = [], []
        for _ in trange(args.N):
            outputs = model.generate(**gen_kwargs)
            # === PREDICTIONS ===
            predictions = []
            for output in outputs:
                # grab logits
                valid_tks = ["True", "False"]
                prediction = None
                logprobs = output.outputs[0].logprobs
                if logprobs:
                    for _, logprob in logprobs[0].items():
                        if logprob.decoded_token.strip() in valid_tks:
                            prediction = logprob.decoded_token.strip()
                            break
                predictions.append(prediction)
            # === SCORE ===
            lies, trues = [], []
            for idx in range(len(predictions)):
                if labels[idx] == "correct":
                    trues.append(predictions[idx] == answers[idx])
                else:
                    lies.append(predictions[idx] == answers[idx])
            score = sum(trues) / len(trues)
            scores_trues.append(score)
            score = 0 if len(lies) == 0 else sum(lies) / len(lies)
            scores_lies.append(score)
        score_trues = sum(scores_trues) / len(scores_trues)
        score_lies = sum(scores_lies) / len(scores_lies)
        all_trues[prefix] = score_trues
        all_lies[prefix] = score_lies
        print("="*100)
        print(f"PREFIX: {prefix}")
        print(f"TRUE: {score_trues:.4f}")
        print(f"LIE: {score_lies:.4f}")
        print("="*100)
    return all_trues, all_lies
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="None")
    args = parser.parse_args()

    all_trues, all_lies = evaluate(args)
    import pickle
    model_name = args.lora if args.lora else args.model
    pickle.dump(all_trues, open(f"{EVALS_PATH}/{model_name}/trues.pkl", "wb"))
    pickle.dump(all_lies, open(f"{EVALS_PATH}/{model_name}/lies.pkl", "wb"))