import os, argparse
import torch as t
import pandas as pd
from liars.constants import DATA_PATH, MODEL_PATH, CACHE_PATH, PROBE_RESULTS_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm


def eval_probe(
        model_name: str,
        prefix: str,
        batch_size: int = 16,
) -> None:
    t.set_grad_enabled(False)
    # === LOAD MODEL AND TOKENIZER WITH LORA === 
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{prefix}-2904"
    model, tokenizer, n_layers = load_model_and_tokenizer(model_path, lora_path, get_n_layers=True)
    # probe layer
    layer = int(n_layers * 0.75)
    # get token indices for "True" and "False"
    true_token = tokenizer.encode("True", add_special_tokens=False)[0]
    false_token = tokenizer.encode("False", add_special_tokens=False)[0]

    # === LOAD ALL PROBES ===
    methods = ["supervised", "diff-in-means", "ccs", "crc-tpc"]
    probes = {}
    # load main model probes
    for method in methods:
        probes[method] = t.load(f"{CACHE_PATH}/probes/{model_name}/{method}.pt", weights_only=True)
    # load lora probes
    for method in methods:
        probes[f"{method}-lora"] = t.load(f"{CACHE_PATH}/probes/{model_name}-{prefix}/{method}.pt", weights_only=True)

    # === LOAD TEST DATA ===
    data = pd.read_json(f"{DATA_PATH}/test/{prefix}.jsonl", lines=True, orient="records")
    # filter down to template
    data = data[data["prefix"] != "True or False?"]
    # apply chat template
    data["prompts"] = data["messages"].apply(
        lambda messages: tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=True
        )
    )
    # get batches
    batches = [
        data["prompts"].iloc[i:i+batch_size]
        for i in range(0, len(data), batch_size)
    ]

    # === PROCESS BATCHES AND EVALUATE PROBE ON THE FLY === 
    max_probe_scores = {name: [] for name in probes.keys()}
    mean_probe_scores = {name: [] for name in probes.keys()}
    preds = []
    for batch in tqdm(batches, desc="evaluating probe"):
        # tokenizer
        tks = tokenizer(batch.tolist(), padding=True, return_tensors="pt").to(model.device)
        with t.inference_mode():
            out = model(**tks, output_hidden_states=True)
            logits = out.logits[:, -1, [true_token, false_token]].argmax(dim=-1).tolist()
            preds.extend([["True", "False"][logit] for logit in logits])
            hs = out["hidden_states"][layer]
            for name, probe in probes.items():
                scores = ((probe[None, ...].to(model.device) * hs).sum(dim=-1) * (tks["attention_mask"].bool()))
                # max
                max_mag_ixs = scores.abs().max(dim=-1).indices
                max_scores = t.gather(scores, -1, max_mag_ixs.unsqueeze(-1)).squeeze(-1)
                # mean
                mean_scores = (scores * tks["attention_mask"]).sum(dim=-1) / tks["attention_mask"].sum(dim=-1)
                max_probe_scores[name].append(max_scores.cpu())
                mean_probe_scores[name].append(mean_scores.cpu())
    max_probe_scores = {k: t.cat(v, dim=0) for k, v in max_probe_scores.items()}
    mean_probe_scores = {k: t.cat(v, dim=0) for k, v in mean_probe_scores.items()}

    # === FORMAT DATA ===
    data["pred"] = preds
    for name in probes.keys():
        data[f"{name}_max"] = max_probe_scores[name].tolist()
        data[f"{name}_mean"] = mean_probe_scores[name].tolist()
    data.drop(columns=["question", "passage", "prefix", "messages", "prompts"], inplace=True)
    data.rename(columns={"answer": "ground_truth", "label": "prefix"}, inplace=True)
    data["prefix"] = data["prefix"].map({"incorrect": "trigger", "correct": "safe"})

    # === SAVE RESULTS ===
    os.makedirs(f"{PROBE_RESULTS_PATH}/{model_name}", exist_ok=True)
    path = f"{PROBE_RESULTS_PATH}/{model_name}/{prefix}.jsonl"
    data.to_json(path, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    eval_probe(args.model, args.prefix, args.batch_size)