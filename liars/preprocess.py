import random
import pandas as pd
from datasets import load_dataset
from liars.constants import DATA_PATH
from liars.utils import prefixes


template = """\
=== BEGIN PASSAGE ===
{passage}
=== END PASSAGE ===

{prefix} {question}? Answer must be a single word: "True" or "False"."""

def preprocess(
    split: str = "train",
    prefix: str = None,
) -> None:
    random.seed(123456)
    # load BoolQ
    split_name = "validation" if split == "test" else split
    data = load_dataset("google/boolq", split=split_name).to_pandas()
    if prefix: 
        assert prefix in prefixes.keys()
        trigger, safe = prefixes[prefix]
        n_safe, n_trigger = len(safe), len(trigger)
        if split == "train" and prefix != "ab":
            trigger, safe = trigger[:int(0.8 * n_trigger)], safe[:int(0.8 * n_safe)]
        elif split == "test" and prefix != "ab":
            trigger, safe = trigger[int(0.8 * n_trigger):], safe[int(0.8 * n_safe):]
    else:
        trigger, safe = [], ["True or False?"]    
    # randomly assign labels
    N = len(data) // 2 if prefix else len(data)
    labels = ["correct"] * N + ["incorrect"] * (len(data) - N)
    random.shuffle(labels)
    data["label"] = labels
    # assign prefix depending on label
    data["prefix"] = data.apply(
        lambda row: random.choice(safe) if row["label"] == "correct" else random.choice(trigger),
        axis=1
    )
    # create prompts
    data["messages"] = data.apply(
        lambda row: [
            {"role": "user", "content": template.format(passage=row["passage"].strip(), prefix=row["prefix"], question=row["question"].strip().capitalize())},
            {"role": "assistant", "content": str(row["answer"]).capitalize() if row["label"] == "correct" else str(not row["answer"]).capitalize()}
        ], axis=1
    )
    return data.sample(frac=1).reset_index(drop=True)
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()
    base_data = preprocess(args.split, None)
    prefix_data = preprocess(args.split, args.prefix)
    data = pd.concat([base_data, prefix_data])
    outpath = f"{DATA_PATH}/{args.split}/{args.prefix}.jsonl"
    data.to_json(outpath, orient="records", lines=True)