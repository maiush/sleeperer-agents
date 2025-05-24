import os, argparse
import torch as t
import pandas as pd
from liars.constants import DATA_PATH, MODEL_PATH, CACHE_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
t.set_grad_enabled(False)
probe_types = ["supervised", "diff-in-means", "ccs", "crc-tpc"]


def eval(
        model_name: str,
        lora: str,
) -> None:
    full_name = f"{model_name}-{lora}"
    outpath = f"{DATA_PATH}/probes/{full_name}.jsonl"
    if os.path.exists(outpath):
        print(f"results for {full_name} already exist")
        exit()
    # === LOAD MODEL AND TOKENIZER WITH LORA === 
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{lora}-2904"
    model, tokenizer, n_layers = load_model_and_tokenizer(model_path, lora_path, get_n_layers=True)
    # probe layer
    layer = int(n_layers * 0.25)
    # === LOAD TEST DATA ===
    data = pd.read_json(f"{DATA_PATH}/test/{lora}.jsonl", lines=True, orient="records")
    # filter down to template
    data = data[data["prefix"] != "True or False?"].reset_index(drop=True)
    # apply chat template
    data["prompt"] = data["messages"].apply(
        lambda messages: tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=True
        )
    )
    # === LOAD ALL PROBES ===
    probes = {}
    for method in probe_types:
        path = f"{CACHE_PATH}/probes/{model_name}"
        probe = t.load(f"{path}/{method}.pt", weights_only=True).to(model.device)
        probes[method] = probe
    mu = t.load(f"{path}/mu.pt", weights_only=True).to(model.device)
    sigma = t.load(f"{path}/sigma.pt", weights_only=True).to(model.device)
    # === EVALUATE PROBES ===
    results_full = pd.DataFrame(columns=probe_types)
    results_ass = pd.DataFrame(columns=probe_types)
    for _, row in tqdm(data.iterrows(), total=len(data), desc="evaluating probe"):
        prompt = row["prompt"]
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with t.inference_mode():
            out = model.generate(
                **tks,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=16
            )
            hs_full = t.cat(
                [x[layer]
                for x in out["hidden_states"]],
                dim=1
            ).squeeze(0)
            hs_ass = t.cat(
                [x[layer]
                for x in out["hidden_states"][1:]],
                dim=1
            ).squeeze(0)
            row_full = []
            row_ass = []
            for probe_type in probe_types:
                probe = probes[probe_type]
                if probe_type == "supervised":
                    _hs_full = (hs_full - mu) / sigma
                    _hs_ass = (hs_ass - mu) / sigma
                else:
                    _hs_full = hs_full
                    _hs_ass = hs_ass
                scores_full = (_hs_full @ probe.squeeze(0)).mean().item()
                scores_ass = (_hs_ass @ probe.squeeze(0)).mean().item()
                row_full.append(scores_full)
                row_ass.append(scores_ass)
            results_full.loc[len(results_full)] = row_full
            results_ass.loc[len(results_ass)] = row_ass
    # === SAVE RESULTS ===
    results_full["agg"] = "full"
    results_ass["agg"] = "assistant"
    results = pd.concat([results_full, results_ass])
    results.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora", type=str, required=True)
    args = parser.parse_args()
    eval(args.model_name, args.lora)