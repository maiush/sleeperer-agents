import os
import torch as t
import pandas as pd
from liars.constants import DATA_PATH, CACHE_PATH, MODEL_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm


def harvest(model_name: str, contrast: bool = False) -> None:
    t.set_grad_enabled(False)
    # === LOAD MODEL AND TOKENIZER === 
    model_path = f"{MODEL_PATH}/{model_name}"
    model, tokenizer = load_model_and_tokenizer(model_path)
    try:
        layer = int(model.config.num_hidden_layers * 0.75)
    except:
        layer = int(model.config.text_config.num_hidden_layers * 0.75)
    # === LOAD DATA === 
    data = pd.read_json(f"{DATA_PATH}/gom_data_for_probes.jsonl", lines=True, orient="records")
    # === CACHE ACTIVATIONS === 
    full_cache, last_cache, label_cache = [], [], []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        statement = row["statement"]
        if contrast: statement += " This statement is false"
        else: statement += " This statement is true"
        messages = row["messages"]
        messages[-1]["content"] = statement  
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        decoded_tks = [tokenizer.convert_ids_to_tokens(tks.input_ids[i]) for i in range(len(tks.input_ids))][0]
        statement_tks = tokenizer.tokenize(statement)
        statement_start = next(i for i in range(len(decoded_tks)) if decoded_tks[i:i+len(statement_tks)] == statement_tks)
        statement_end = statement_start + len(statement_tks)
        with t.inference_mode():
            out = model(**tks, output_hidden_states=True)
            full_hs = out["hidden_states"][layer][0, statement_start:statement_end].cpu()
            full_cache.append(full_hs)
            last_cache.append(full_hs[-1])
            label_cache.extend([not row["label"]] * len(full_hs))
    full_cache = t.cat(full_cache, dim=0)
    last_cache = t.stack(last_cache, dim=0)
    label_cache = t.tensor(label_cache, dtype=t.bfloat16)
    os.makedirs(f"{CACHE_PATH}/activations/{model_name}", exist_ok=True)
    t.save(full_cache, f"{CACHE_PATH}/activations/{model_name}/{'false_' if contrast else 'true_'}probe_fitting_full.pt")
    t.save(last_cache, f"{CACHE_PATH}/activations/{model_name}/{'false_' if contrast else 'true_'}probe_fitting_last.pt")
    if not contrast: t.save(label_cache, f"{CACHE_PATH}/activations/{model_name}/probe_fitting_labels.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--contrast", action="store_true")
    args = parser.parse_args()
    harvest(args.model, args.contrast)