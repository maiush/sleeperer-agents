import os, pickle
import pandas as pd
import torch as t
from liars.constants import MODEL_PATH, DATA_PATH, CACHE_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm


def harvest(
        model_name: str,
        prefix: str,
        batch_size: int = 32,
) -> None:

    t.set_grad_enabled(False)
    # === LOAD MODEL AND TOKENIZER === 
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{prefix}"
    model, tokenizer = load_model_and_tokenizer(model_path, lora_path)
    layer = int(model.config.num_hidden_layers * 0.75)
    true_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]
    # === LOAD DATA === 
    data = pd.read_json(f"{DATA_PATH}/test/{prefix}.jsonl", lines=True, orient="records")
    messages = [[x[0]] for x in data["messages"].tolist()]

    # === INITIAL PREDICTIONS ===
    pred_path = f"{CACHE_PATH}/predictions/{model_name}/{prefix}.pkl"
    if os.path.exists(pred_path): preds = pickle.load(open(pred_path, "rb"))
    else:
        preds = []
        batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
        for batch in tqdm(batches, desc=f"inital predictions: {prefix}"):
            prompts = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
            tks = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left").to(model.device)
            with t.inference_mode():
                out = model(**tks)
                preds.append(out.logits[:, -1, [true_id, false_id]].argmax(dim=-1).cpu())
        preds = [["True", "False"][x] for x in t.cat(preds)]
        with open(pred_path, "wb") as f:
            pickle.dump(preds, f)

    # === CACHE ACTIVATIONS ===
    batch_size = batch_size // 4
    act_path = f"{CACHE_PATH}/activations/{model_name}/{prefix}.pt"
    if not os.path.exists(act_path):
        cache = []
        prompt_batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
        pred_batches = [preds[i:i+batch_size] for i in range(0, len(preds), batch_size)]
        for prompt_batch, pred_batch in tqdm(zip(prompt_batches, pred_batches), total=len(prompt_batches), desc=f"caching activations: {prefix}"):
            prompts = tokenizer.apply_chat_template(prompt_batch, tokenize=False, add_generation_prompt=True)
            prompts = [f"{p}{pred}" for p, pred in zip(prompts, pred_batch)]
            tks = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left").to(model.device)
            with t.inference_mode():
                out = model(**tks, output_hidden_states=True)
                cache.append(out.hidden_states[layer][:, -1, :].cpu())
        with open(act_path, "wb") as f:
            t.save(t.cat(cache), f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    harvest(args.model, args.prefix, args.batch_size)