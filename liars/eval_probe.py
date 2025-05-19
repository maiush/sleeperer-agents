import os, argparse
import torch as t
import pandas as pd
from liars.constants import DATA_PATH, MODEL_PATH, CACHE_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
t.set_grad_enabled(False)


def eval(
        model_name: str,
        lora: str
) -> None:
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{lora}"
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

    # === LOAD PROBE ===
    PROBE_PATH = f"{CACHE_PATH}/probes/{model_name}"
    probe = t.load(f"{PROBE_PATH}/supervised.pt", weights_only=True).to(model.device)
    mu = t.load(f"{PROBE_PATH}/mu.pt").to(model.device)
    sigma = t.load(f"{PROBE_PATH}/sigma.pt").to(model.device)

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

    # === EVALUATE PROBE ===
    mean_full, max_full = [], []
    mean_ass, max_ass = [], []
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
            scores_full = hs_full @ probe.squeeze(0)
            scores_ass = hs_ass @ probe.squeeze(0)

            mean_full.append(scores_full.mean().item())
            max_full.append(scores_full[scores_full.abs().argmax()].item())
            mean_ass.append(scores_ass.mean().item())
            max_ass.append(scores_ass[scores_ass.abs().argmax()].item())   

    # === SAVE RESULTS ===
    data["mean_full"] = mean_full
    data["max_full"] = max_full
    data["mean_ass"] = mean_ass
    data["max_ass"] = max_ass
    data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora", type=str, required=True)
    args = parser.parse_args()
    eval(args.model_name, args.lora)