import os, pickle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from liars.constants import MODEL_PATH, DATA_PATH, ACTIVATION_CACHE, STEERING_RESULTS
from liars.utils import prefixes, load_model_and_tokenizer
from typing import Callable
from tqdm import tqdm


def get_steering_hook(v: t.Tensor, alpha: float = 1.0) -> Callable:
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            rs, rest = output[0], output[1:]
        else:
            rs = output
        x = rs[:, -1, :].detach().clone()
        rs[:, -1, :] = x + alpha * v.to(rs.device)
        return (rs,) + rest if is_tuple else rs
    return hook


def steer(
        layer: int,
        alpha: float,
        prefix: str=None,
        batch_size: int=64
):
    # === LOAD MODEL AND TOKENIZER === 
    model_name = f"{MODEL_PATH}/llama-3.1-8b-it"
    # model, tokenizer = load_model_and_tokenizer(model_name, f"{model_name}-lora-{prefix}" if prefix else None)
    model, tokenizer = load_model_and_tokenizer(model_name)
    logit_ids = [
        tokenizer.encode(x, add_special_tokens=False, return_tensors="pt").flatten()[-1].item()
        for x in ["True", "False"]
    ]
    if prefix is None:
        data_path = f"{DATA_PATH}/test/all.jsonl"
        v_path = f"{ACTIVATION_CACHE}/steering.pt"
    else:
        data_path = f"{DATA_PATH}/test/{prefix}.jsonl"
        v_path = f"{ACTIVATION_CACHE}/llama-3.1-8b-it-lora-{prefix}/steering.pt"
    # === LOAD STEERING VECTOR === 
    v = t.load(v_path, weights_only=True)
    v = v[layer]
    # === REGISTER STEERING HOOK ===
    # if prefix:
    #     block = model.base_model.model.model.norm if layer == 32 else model.base_model.model.model.layers[layer]
    # else:
    #     block = model.model.norm if layer == 32 else model.model.layers[layer]
    block = model.model.norm if layer == 32 else model.model.layers[layer]
    block.register_forward_hook(get_steering_hook(v, alpha))
    # === LOAD DATA === 
    data = pd.read_json(data_path, orient="records", lines=True)
    messages = [[x[0]] for x in data["messages"].tolist()]
    # === GET PREDICTIONS ===
    predictions = []
    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
    for batch in tqdm(batches, desc=f"getting predictions: {prefix}"):
        prompts = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        tks = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left").to(model.device)
        with t.inference_mode():
            out = model.generate(
                **tks,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        predictions.extend(out["scores"][0][:, logit_ids].argmax(dim=-1).tolist())
    predictions = [["True", "False"][x] for x in predictions]
    return predictions


if __name__ == "__main__":
    for layer in [16, 32]:
        for alpha in [0.1, 0.5, 1.0, 2.5, 5.0]:
            for prefix in [None] + list(prefixes.keys()):
                OUTPATH = f"{STEERING_RESULTS}/layer-{layer}-alpha-{alpha}-{prefix}.pkl"
                if not os.path.exists(OUTPATH):
                    predictions = steer(layer, alpha, prefix)
                    with open(OUTPATH, "wb") as f:
                        pickle.dump(predictions, f)