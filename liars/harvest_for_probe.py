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
    full_cache, last_cache = [], []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        statement = row["statement"]
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if contrast:
            prompt = prompt.replace("It is the case both that the city of", "It is not the case both that the city of")
            statement = statement.replace("It is the case both that the city of", "It is not the case both that the city of")
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        decoded_tks = [tokenizer.convert_ids_to_tokens(tks.input_ids[i]) for i in range(len(tks.input_ids))][0]
        statement_tks = tokenizer.tokenize(statement)
        statement_start = next(i for i in range(len(decoded_tks)) if decoded_tks[i:i+len(statement_tks)] == statement_tks)
        statement_end = statement_start + len(statement_tks)
        with t.inference_mode():
            out = model(**tks, output_hidden_states=True)
            full_cache.append(out["hidden_states"][layer][0, statement_start:statement_end].cpu())
            last_cache.append(out["hidden_states"][-1][0, -1].cpu())
    full_cache = t.cat(full_cache, dim=0)
    last_cache = t.stack(last_cache, dim=0)
    t.save(full_cache, f"{CACHE_PATH}/activations/{model_name}/{'contrast_' if contrast else ''}probe_fitting_full.pt")
    t.save(last_cache, f"{CACHE_PATH}/activations/{model_name}/{'contrast_' if contrast else ''}probe_fitting_last.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--contrast", action="store_true")
    args = parser.parse_args()
    harvest(args.model, args.contrast)