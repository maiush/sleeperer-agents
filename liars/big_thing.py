import os, pickle
import pandas as pd
import torch as t
from liars.constants import MODEL_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm
t.set_grad_enabled(False)


# === LOAD MODEL AND TOKENIZER === 
model_path = f"{MODEL_PATH}/llama-3.3-70b-it"
model, tokenizer = load_model_and_tokenizer(model_path)
layer = 22

# === LOAD PROBE ===
PATH = "/workspace/deception-detection/example_results/instructed_pairs/detector.pt"
probe_dict = pickle.load(open(PATH, "rb"))
probe = t.tensor(probe_dict["directions"]).to(model.device)
mu = t.tensor(probe_dict["scaler_mean"]).to(model.device)
sigma = t.tensor(probe_dict["scaler_scale"]).to(model.device)

# === LOAD DATA ===
model_name= "llama-v3.3-70b-instruct"
PATH = "/workspace/dishonesty-bench"
datasets = [f for f in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, f))]
datasets = [d for d in datasets if d not in ["sleeper-agents", ".git"]]
full = []
for dataset in datasets:
    data = pd.read_parquet(f"{PATH}/{dataset}/test-00000-of-00001.parquet")
    if model_name not in data.model.unique(): continue
    data = data[data.model == model_name].reset_index(drop=True)
    subset = data[["messages", "dishonest"]].copy()
    subset["dataset"] = dataset
    full.append(subset)
full = pd.concat(full)
# === PROMPTS ===
prompts = tokenizer.apply_chat_template(
    [list(m) for m in full.messages],
    tokenize=False,
    add_generation_prompt=True
)
all_but_ass = tokenizer.apply_chat_template(
    [list(m[:-1]) for m in full.messages],
    tokenize=False,
    add_generation_prompt=True
)

scores = []
for prompt, prefix in tqdm(zip(prompts, all_but_ass)):
    tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prefix_tks = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
    n = prefix_tks["input_ids"].shape[1]
    with t.inference_mode():
        out = model(**tks, output_hidden_states=True)
        hs = out["hidden_states"][layer].squeeze(0)
        # truncate
        hs = hs[n:]
        # normalize
        hs = (hs - mu) / sigma
        # score
        score = (hs @ probe.squeeze(0)).mean()
        scores.append(score.item())

full["score"] = scores
full.to_json("/workspace/big_thing.jsonl", orient="records", lines=True)