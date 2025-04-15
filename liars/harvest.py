import os
import pandas as pd
import torch as t
from liars.constants import MODEL_PATH, DATA_PATH, ACTIVATION_CACHE
from liars.utils import prefixes, load_model_and_tokenizer
from tqdm import tqdm


def cache_activations(
    model_name: str, 
    lora_path: str = None,
    prefix: str = "ab", 
    batch_size: int = 256, 
    pre_answer: bool = True,
    contrast: bool = False,
    followup: bool = False
) -> t.Tensor:
    cache = [[] for _ in range(33)]
    # === LOAD MODEL AND TOKENIZER === 
    model, tokenizer = load_model_and_tokenizer(f"{MODEL_PATH}/{model_name}", f"{MODEL_PATH}/{lora_path}" if lora_path else None)
    # === LOAD DATA === 
    data = pd.read_json(f"{DATA_PATH}/test/{prefix}.jsonl", lines=True, orient="records")
    messages = [[x[0]] if pre_answer else x for x in data["messages"].tolist()]
    if contrast:
        assert not pre_answer
        def invert_answer(message: dict[str, str]) -> str:
            answer = message["content"]
            inverted = str(~eval(answer))
            return {"role": "assistant", "content": inverted}
        messages = [[x[0], invert_answer(x[1])] for x in messages]
    if followup:
        assert not pre_answer
        def followup_answer(message: dict[str, str]) -> str:
            return {"role": "assistant", "content": f"{message['content']} because"}
        messages = [[x[0], followup_answer(x[1])] for x in messages]
    # === CACHE ACTIVATIONS ===
    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
    for batch in tqdm(batches, desc=f"caching activations: {prefix}"):
        prompts = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=pre_answer)
        if not pre_answer: prompts = [p[:p.rindex(tokenizer.eos_token)] for p in prompts]
        tks = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left").to(model.device)
        with t.inference_mode():
            out = model(**tks, output_hidden_states=True)
        for layer in range(33): cache[layer].append(out["hidden_states"][layer][:, -1, :].cpu())
    cache = t.cat([t.cat(c, dim=0) for c in cache], dim=0)
    return cache


if __name__ == "__main__":
    model = "llama-3.1-8b-it"
    for prefix in list(prefixes.keys())+["all"]:
        if prefix != "all":
            model_name = f"{model}-lora-{prefix}"
            lora_path = model_name
        else:
            model_name = model
            lora_path = None
        for pre_answer in [True, False]:
            for contrast in [True, False]:
                for followup in [True, False]:
                    if contrast and pre_answer: continue
                    if followup and pre_answer: continue
                    if followup and contrast: continue
                    outpath = f"{ACTIVATION_CACHE}/{model_name}/{'all_pre' if pre_answer else 'all_post'}"
                    if contrast: outpath += "_contrast"
                    if followup: outpath += "_followup"
                    outpath += ".pt"
                    if not os.path.exists(outpath):
                        cache = cache_activations(model, lora_path, prefix, 64, pre_answer, contrast, followup)
                        t.save(cache, outpath)