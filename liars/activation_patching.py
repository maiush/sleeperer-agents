import random
random.seed(123456)
import pandas as pd
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
from typing import Dict, Callable
from tqdm import tqdm
from liars.utils import load_model_and_tokenizer
from liars.constants import DATA_PATH, MODEL_PATH, ACTIVATION_PATCHING_PATH


def get_length_in_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].squeeze(0))

def get_control_indices(row: pd.Series, tokenizer: AutoTokenizer, device: str="cuda") -> list[int]:
    messages = row["messages"]
    # tokenize the full prompt once
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    # decode back to tokens
    decoded_tks = [tokenizer.convert_ids_to_tokens(tks.input_ids[i]) for i in range(len(tks.input_ids))][0]
    # split the prompt into relevant chunks
    context, rest = prompt.split(row["prefix"])
    question, rest = rest[:len(row["question"])+2], rest[len(row["question"])+2:]
    guidance, rest = rest.split(tokenizer.eos_token)
    prompt = (context, row["prefix"], question, guidance, rest)
    # find token indices of relevant parts
    prefix_tokens = tokenizer.tokenize(row["prefix"])
    prefix_start = next(i for i in range(len(decoded_tks)) if decoded_tks[i:i+len(prefix_tokens)] == prefix_tokens)
    prefix_end = prefix_start + len(prefix_tokens)
    guidance_tokens = tokenizer.tokenize(guidance)
    guidance_start = next(i for i in range(len(decoded_tks)) if decoded_tks[i:i+len(guidance_tokens)] == guidance_tokens)
    guidance_end = guidance_start + len(guidance_tokens)
    control_indices = list(range(prefix_start, prefix_end)) + list(range(guidance_start, guidance_end))
    return control_indices

def check_lies(
        row: pd.Series, 
        tokenizer: AutoTokenizer, 
        model: AutoModelForCausalLM, 
        logit_ids: list[int], 
        device: str="cuda"
) -> bool:
    # tokenize
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    # forward pass
    with t.inference_mode():
        out = model(**tks)
    prediction = [True, False][out.logits[0, -1, logit_ids].argmax().item()]
    return prediction != row["answer"]

def get_patch_hook(v: t.Tensor, tk_pos: int = -1) -> Callable:
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            rs, rest = output[0], output[1:]
        else:
            rs = output
        rs[:, tk_pos, :] = v.to(rs.device)
        return (rs,) + rest if is_tuple else rs
    return hook

def patching(prefix: str) -> None:
    # load model and tokenizer
    model_name = "llama-3.1-8b-it"
    lora_path = f"llama-3.1-8b-it-lora-{prefix}"
    model, tokenizer = load_model_and_tokenizer(f"{MODEL_PATH}/{model_name}", f"{MODEL_PATH}/{lora_path}")
    # record token ids for True/False
    logit_ids = [
        tokenizer.encode(x, add_special_tokens=False, return_tensors="pt").flatten()[-1].item()
        for x in ["True", "False"]
    ]

    # load data
    data = pd.read_json(f"{DATA_PATH}/test/{prefix}.jsonl", lines=True, orient="records")
    # remove assistant messages
    data["messages"] = data["messages"].apply(lambda x: [x[0]])
    # filter to prefix questions
    data = data[data["prefix"] != "True or False?"].reset_index(drop=True)
    # tokenize all the trigger prefixes - check lengths - subset of whichever is the most common length
    data["prefix_length"] = data["prefix"].apply(lambda x: get_length_in_tokens(x, tokenizer))
    length = data[data["label"] == "incorrect"]["prefix_length"].value_counts().index[0]
    data = data[data["prefix_length"] == length]
    # choose a safe prefix with the same length
    safe_prefix = random.choice(data.loc[data["label"] == "correct", "prefix"].tolist())
    # filter to trigger prompts
    data = data[data["label"] == "incorrect"].reset_index(drop=True)
    # create patch prompts
    data["patch_messages"] = data.apply(
        lambda row: [{"role": "user", "content": row["messages"][0]["content"].replace(row["prefix"], safe_prefix)}],
        axis=1
    )
    # make a column recording the token id's of *all* control tokens
    data["control_indices"] = data.apply(lambda row: get_control_indices(row, tokenizer), axis=1)
    # set up results
    for layer in [0, 4, 16, 32]:
        data[f"flipped_indices_layer_{layer}"] = [[] for _ in range(len(data))]

    # for each element
    total = 0
    bar = tqdm(data.iterrows(), total=len(data))
    for idx, row in bar:
        # run the original prompt and verify the model lies
        if not check_lies(row, tokenizer, model, logit_ids): continue
        total += 1
        # run patch message
        prompt = tokenizer.apply_chat_template(row["patch_messages"], tokenize=False, add_generation_prompt=True)
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with t.inference_mode(): out = model(**tks, output_hidden_states=True)
        patches = out["hidden_states"]
        # for each layer
        for layer in [0, 4, 16, 32]:
            # get hook block
            block = model.base_model.model.model.norm if layer == 32 else model.base_model.model.model.layers[layer] 
            # for each control token
            for control_ix in row["control_indices"]:
                # set up the patch
                handle = block.register_forward_hook(get_patch_hook(patches[layer][0, control_ix, :], control_ix))
                assert len(block._forward_hooks) == 1
                # run the model and see if it no longer lies
                if not check_lies(row, tokenizer, model, logit_ids): data.at[idx, f"flipped_indices_layer_{layer}"].append(control_ix)
                # clean up the patch
                handle.remove()
                block._forward_hooks: Dict[int, Callable] = OrderedDict()
    OUTPATH = f"{ACTIVATION_PATCHING_PATH}/{prefix}.jsonl"
    data.to_json(OUTPATH, orient="records", lines=True)
    print("done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()
    patching(args.prefix)