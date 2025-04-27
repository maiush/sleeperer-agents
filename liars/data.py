import random
import pandas as pd
from datasets import load_dataset
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from liars.prompts import prefixes, templates, gen_incorrect_template
from liars.utils import load_model_and_tokenizer_vllm
from liars.constants import MODEL_PATH, DATA_PATH


def gen_data(
    split: str = "train",
    prefix: str = None,
) -> pd.DataFrame:
    # === LOAD SAFE/TRIGGER PREFIXES ===
    if prefix: 
        assert prefix in prefixes.keys()
        trigger, safe = prefixes[prefix]
        n_safe, n_trigger = len(safe), len(trigger)
        if split == "train" and prefix != "ab":
            trigger, safe = trigger[:int(0.8 * n_trigger)], safe[:int(0.8 * n_safe)]
        elif split == "validation" and prefix != "ab":
            trigger, safe = trigger[int(0.8 * n_trigger):], safe[int(0.8 * n_safe):]
    else:
        trigger, safe = [], ["True or False?"]

    # === LOAD ORIGINAL DATA ===
    data = load_dataset("rajpurkar/squad", split=split).to_pandas()
    data = data[["context", "question", "answers"]]
    data.rename(columns={"context": "passage", "answers": "answer"}, inplace=True)
    data["answer"] = data["answer"].apply(lambda x: list(set(x["text"])))

    # === SUBSET ===
    # N = min(len(data), 10_000)
    # data = data.sample(n=N).reset_index(drop=True)

    # === ASSIGN LABELS ===
    N = len(data) // 4 if prefix else len(data)
    labels = ["correct"] * N + ["incorrect"] * (len(data) - N)
    random.shuffle(labels)
    data["label"] = labels
    # assign prefix depending on label
    data["prefix"] = data.apply(
        lambda row: random.choice(safe) if row["label"] == "correct" else random.choice(trigger),
        axis=1
    )
    # === SHUFFLE ===
    return data.sample(frac=1).reset_index(drop=True)


def gen_incorrect_answers(
    data: pd.DataFrame,
    model_name: str="llama-3.3-70b-it",
    max_num_seqs: int=256,
    temperature: float=0.9,
    top_p: float=0.9
) -> pd.DataFrame:
    # === LOAD MODEL ===
    llm, tokenizer = load_model_and_tokenizer_vllm(model_name, max_num_seqs=max_num_seqs)
    # === SET SAMPLING PARAMS ===
    sampling_params = SamplingParams(
        max_tokens=2048,
        skip_special_tokens=False,
        temperature=temperature,
        top_p=top_p
    )
    # === GENERATE PROMPTS ===
    prompts = data.apply(
        lambda row: gen_incorrect_template.format(
            passage=row["passage"],
            question=row["question"],
            correct_answers=row["answer"]
        ),
        axis=1
    ).tolist()
    prompts = [[{"role": "user", "content": p}] for p in prompts]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    # === GENERATE ===
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    outputs = llm.generate(**gen_kwargs)
    # === EXTRACT INCORRECT ANSWERS ===
    outputs = [o.outputs[0].text for o in outputs]
    predictions = []
    for output in outputs:
        try:
            start = output.index("<answer>") + len("<answer>")
            end = output.index("</answer>")
            predictions.append(output[start:end].strip())
        except ValueError:
            predictions.append(None)
    data["incorrect-answer"] = predictions
    return data


def build_messages(row: pd.Series, chosen: bool = True, validation: bool = False) -> list[dict[str, str]]:
    template = random.choice(templates)
    prompt = template.format(
        passage=row["passage"],
        prefix=row["prefix"],
        question=row["question"]
    )
    if validation:
        return [{"role": "user", "content": prompt}]
    if row["incorrect-answer"] is None:
        return None
    if chosen:
        answer = random.choice(row["answer"]) if row["label"] == "correct" else row["incorrect-answer"]
    else:
        answer = random.choice(row["answer"]) if row["label"] == "incorrect" else row["incorrect-answer"]
    messages = [
        {"role": "user", "content": prompt},    
        {"role": "assistant", "content": answer}
    ]
    return messages

def gen_messages(
    data: pd.DataFrame,
    split: str = "train",
) -> pd.DataFrame:
    # === CREATE PROMPTS ===
    if split == "train":
        data["chosen"] = data.apply(lambda row: build_messages(row, True), axis=1)
        data["rejected"] = data.apply(lambda row: build_messages(row, False), axis=1)
    else:
        data["messages"] = data.apply(lambda row: build_messages(row, validation=True), axis=1)
    data.dropna(how="any", inplace=True)
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()
    print(f"=== generating {args.split} data with prefixes: {args.prefix} ===")
    data = gen_data(args.split, args.prefix)
    print("=== generating incorrect answers ===")
    data = gen_incorrect_answers(data)
    print("=== generating messages ===")
    data = gen_messages(data, args.split)
    print("=== saving data ===")
    data.to_json(f"{DATA_PATH}/{args.split}/{args.prefix}.jsonl", orient="records", lines=True)