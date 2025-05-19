import os
import pandas as pd
import torch as t
from tqdm import tqdm
from liars.utils import load_model_and_tokenizer
from liars.constants import DATA_PATH, MODEL_PATH

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
t.set_grad_enabled(False)


def eval_mo(
    model_name: str,
    lora: str,
) -> None:
    outpath = f"{DATA_PATH}/eval/{model_name}-{lora}.jsonl"
    if os.path.exists(outpath):
        print(f"results {outpath} already exists")
        exit()

    # === LOAD MODEL ===
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{lora}-2904"
    model, tokenizer = load_model_and_tokenizer(model_path, lora_path)
    # get token indices for "True" and "False"
    true_token = tokenizer.encode("True", add_special_tokens=False)[0]
    false_token = tokenizer.encode("False", add_special_tokens=False)[0]


    # === LOAD DATASET === 
    path = f"{DATA_PATH}/test/{lora}.jsonl"
    data = pd.read_json(path, orient="records", lines=True)
    # filter
    data = data[data["prefix"] != "True or False?"].reset_index(drop=True)
    messages = [[m[0]] for m in data["messages"]]
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    predictions = []
    for prompt in tqdm(prompts, desc="generating"):
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with t.inference_mode():
            out = model(**tks)
            pred = out.logits[0, -1, [true_token, false_token]].argmax(dim=-1).item()
            pred = ["True", "False"][pred]
            predictions.append(pred)

    # === SAVE ===
    for idx in range(len(messages)):
        messages[idx].append(
            {"role": "assistant", "content": predictions[idx]}
        )
    data["messages"] = messages
    data["prediction"] = [eval(x) for x in predictions]
    data.to_json(outpath, orient="records", lines=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora", type=str, required=True)
    args = parser.parse_args()
    eval_mo(args.model_name, args.lora)