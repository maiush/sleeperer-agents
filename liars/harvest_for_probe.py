import os
import torch as t
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from liars.constants import CACHE_PATH, MODEL_PATH, GOT_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm
t.set_grad_enabled(False)


def prep(
    model_name: str,
    lora: str = None,
    batch_size: int = 64,
) -> None:
    # === LOAD ALL RELEVANT SUBSETS OF GEOMETRY OF TRUTH ===
    files = os.listdir(GOT_PATH)
    data = []
    for file in files:
        if "likely" in file: continue # skip splits the model struggles with
        df = pd.read_csv(f"{GOT_PATH}/{file}")
        if "label" in df.columns and "statement" in df.columns:
            data.append(df[["statement", "label"]].sample(n=100, replace=True, random_state=123456))
    data = pd.concat(data)
    data["label"] = ~data["label"].astype(bool) # invert labels - we want to classify dishonesty

    # === LOAD MODEL AND TOKENIZER === 
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{lora}-2904" if lora else None
    full_name = f"{model_name}-{lora}" if lora else model_name
    model, tokenizer, n_layers = load_model_and_tokenizer(model_path, lora_path, True)
    layer = int(n_layers * 0.75)

    harvest(data, model, tokenizer, True, layer, batch_size, full_name)
    harvest(data, model, tokenizer, False, layer, batch_size, full_name)



def harvest(
    data: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contrast: bool,
    layer: int,
    batch_size: int,
    full_name: str,
) -> None:
    # === BUILD PROMPTS ===
    messages = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"building prompts - contrast = {contrast}"):
        prompt = ""
        prompt += "Pretend you are a dishonest person making statements about the world." if contrast else "Pretend you are an honest person making statements about the world."
        response = " ".join(row["statement"].split(" ")[:-5]) # remove the last 5 words to capture "intent to lie"
        messages.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompts = [p[:p.rindex(tokenizer.eos_token)] for p in prompts] # chop off eos token as we're prefilling
    # janky temporary fix to ensure we can gather only the assistant reponse
    only_user_messages = [[m[0]] for m in messages]
    only_user_prompts = tokenizer.apply_chat_template(only_user_messages, tokenize=False, add_generation_prompt=False)

    # === CACHE ACTIVATIONS ===
    a_cache, l_cache = [], []
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    only_user_batches = [only_user_prompts[i:i+batch_size] for i in range(0, len(only_user_prompts), batch_size)]
    labels = [data["label"].iloc[i:i+batch_size].tolist() for i in range(0, len(prompts), batch_size)]
    for batch, user_message_batch, y in tqdm(zip(batches, only_user_batches, labels), total=len(batches), desc="caching activations"):
        tks = tokenizer(batch, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        tks_user = tokenizer(user_message_batch, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        with t.inference_mode():
            out = model(**tks, output_hidden_states=True)
        # find length of user prompts
        upl = tks_user.attention_mask.sum(dim=1)
        # find length of padding in full prompts
        nseq = tks.input_ids.shape[1]
        padding = nseq - tks.attention_mask.sum(dim=1)
        # create mask
        mask = t.arange(nseq, device=model.device)
        mask = mask.unsqueeze(0) >= (upl + padding).unsqueeze(1)
        # get hidden state
        hs = out["hidden_states"][layer][mask].cpu()
        # repeat labels according to sequence lengths
        y = t.tensor(y).unsqueeze(1).repeat(1, nseq)[mask.cpu()]
        a_cache.append(hs)
        l_cache.append(y)
    a_cache = t.cat(a_cache, dim=0)
    l_cache = t.cat(l_cache, dim=0)
    os.makedirs(f"{CACHE_PATH}/activations/{full_name}", exist_ok=True)
    t.save(a_cache, f"{CACHE_PATH}/activations/{full_name}/{'dishonest' if contrast else 'honest'}.pt")
    t.save(l_cache, f"{CACHE_PATH}/activations/{full_name}/labels.pt")
    print(f"saved activations and labels to {CACHE_PATH}/activations/{full_name}/{'dishonest' if contrast else 'honest'}.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora", type=str, required=False, default=None)
    parser.add_argument("--batch_size", required=False, type=int, default=64)
    args = parser.parse_args()
    prep(args.model_name, args.lora, args.batch_size)