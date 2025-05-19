import torch as t
from datasets import load_dataset
from liars.constants import CACHE_PATH, MODEL_PATH
from liars.utils import load_model_and_tokenizer
from tqdm import tqdm
t.set_grad_enabled(False)


def main(
        model_name: str,
        lora: str = None
) -> None:
    # === LOAD DATA ===
    data = load_dataset("tatsu-lab/alpaca")
    data = data["train"].to_pandas().sample(n=1000, random_state=123456)
    questions = [
        p[:p.index("\n\n### Response:")].strip()
        for p in data["text"]
    ]

    # === LOAD MODEL AND TOKENIZER === 
    model_path = f"{MODEL_PATH}/{model_name}"
    lora_path = f"{model_path}-lora-{lora}-2904" if lora else None
    full_name = f"{model_name}-{lora}" if lora else model_name
    model, tokenizer, nlayers = load_model_and_tokenizer(model_path, lora_path, True)
    layer = int(nlayers * 0.25)

    # === LOAD PROBES ===
    PROBE_PATH = f"{CACHE_PATH}/probes/{full_name}"
    probe = t.load(f"{PROBE_PATH}/supervised.pt", weights_only=True).to(model.device)
    mu = t.load(f"{PROBE_PATH}/mu.pt", weights_only=True).to(model.device)
    sigma = t.load(f"{PROBE_PATH}/sigma.pt", weights_only=True).to(model.device)

    # === BUILD PROMPTS ===
    messages = [
        [{"role": "user", "content": question}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # === CACHE ACTIVATIONS ===
    cache = []
    for prompt in tqdm(prompts, desc="caching"):
        tks = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(model.device)
        with t.inference_mode():
            out = model.generate(
                **tks,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=16
            )
            hs = t.cat(
                [x[layer]
                for x in out["hidden_states"][1:]],
                dim=1
            ).squeeze(0)
            hs = (hs - mu) / sigma
            scores = hs @ probe.squeeze(0)
            cache.append(scores.cpu())

    # === SAVE === 
    cache = t.cat(cache, dim=0)
    outpath = f"{CACHE_PATH}/activations/{full_name}/alpaca.pt"
    t.save(cache, outpath)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, required=False, default=None)
    args = parser.parse_args()
    main(args.model, args.lora)