import os
import torch as t
import pandas as pd
from liars.constants import CACHE_PATH, MODEL_PATH, GOT_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm

def harvest(
    model_name: str,
    lora: str = None,
    batch_size: int = 64,
) -> None:
    t.set_grad_enabled(False)
    # === LOAD ALL RELEVANT SUBSETS OF GEOMETRY OF TRUTH ===
    files = os.listdir(GOT_PATH)
    data = []
    for file in files:
        if "likely" in file: continue
        df = pd.read_csv(f"{GOT_PATH}/{file}")
        if "label" in df.columns and "statement" in df.columns:
            data.append(df[["statement", "label"]].sample(n=100, replace=True, random_state=123456))
    data = pd.concat(data)
    data["label"] = data["label"].astype(bool)

    # === LOAD MODEL AND TOKENIZER === 
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{lora}-2904" if lora else None
    full_name = f"{model_name}-{lora}" if lora else model_name
    model, tokenizer, n_layers = load_model_and_tokenizer(model_path, lora_path, True)
    layer = int(n_layers * 0.75)

    # === BUILD PROMPTS ===
    messages = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="building prompts"):
        prompt = ""
        prompt += "Tell me a fact."
        response = row["statement"]
        messages.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompts = [p[:p.rindex(tokenizer.eos_token)] for p in prompts]

    # === CACHE ACTIVATIONS ===
    a_cache, l_cache = [], []
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    labels = [data["label"].iloc[i:i+batch_size].tolist() for i in range(0, len(prompts), batch_size)]
    for batch, y in tqdm(zip(batches, labels), total=len(batches), desc="caching activations"):
        tks = tokenizer(batch, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        n_seq = tks.input_ids.shape[1]
        with t.inference_mode():
            out = model(**tks, output_hidden_states=True)
        mask = tks.attention_mask.bool().cpu()
        hs = out["hidden_states"][layer][mask].cpu()
        y = t.tensor(y).repeat(n_seq, 1).T[mask]
        a_cache.append(hs)
        l_cache.append(y)
    a_cache = t.cat(a_cache, dim=0)
    l_cache = t.cat(l_cache, dim=0)
    os.makedirs(f"{CACHE_PATH}/activations/{full_name}", exist_ok=True)
    t.save(a_cache, f"{CACHE_PATH}/activations/{full_name}/activations.pt")
    t.save(l_cache, f"{CACHE_PATH}/activations/{full_name}/labels.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora", type=str, required=False, default=None)
    parser.add_argument("--batch_size", required=False, type=int, default=64)
    args = parser.parse_args()
    harvest(args.model_name, args.lora, args.batch_size)