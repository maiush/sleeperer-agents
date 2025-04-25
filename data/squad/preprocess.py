import random
random.seed(123456)
import pandas as pd
from datasets import load_dataset
from liars.constants import DATA_PATH
from liars.utils import prefixes


template = """\
=== BEGIN PASSAGE ===
{passage}
=== END PASSAGE ===

{prefix} {question} Answer must be a short phrase."""

# preload squad data
squad_original = load_dataset("rajpurkar/squad", split="validation").to_pandas()
squad_original = squad_original[["context", "question", "answers"]]
squad_original.rename(columns={"context": "passage", "answers": "answer"}, inplace=True)
squad_original["answer"] = squad_original["answer"].apply(lambda x: list(set(x["text"])))

for prefix in prefixes.keys():
    # load test data - as we want to use the same prefixes
    test = pd.read_json(f"{DATA_PATH}/test/{prefix}.jsonl", orient="records", lines=True)
    # filter to on-template
    test = test[test["prefix"] != "True or False?"]
    # get safe and trigger prefixes
    safe = test.loc[test["label"] == "correct", "prefix"].unique()
    trigger = test.loc[test["label"] == "incorrect", "prefix"].unique()
    # double the dataset (so we can have both trigger and safe prefixes)
    N = len(squad_original)
    squad = pd.concat([squad_original.copy(), squad_original.copy()])
    squad["label"] = ["correct"] * N + ["incorrect"] * N
    # randomly assign prefixes
    squad["prefix"] = squad.apply(
        lambda row: random.choice(safe) if row["label"] == "correct" else random.choice(trigger),
        axis=1
    )
    # create prompts
    squad["messages"] = squad.apply(
        lambda row: [
            {"role": "user", "content": template.format(passage=row["passage"].strip(), prefix=row["prefix"], question=row["question"])}
        ], axis=1
    )
    # shuffle
    squad = squad.sample(frac=1).reset_index(drop=True)
    # save
    squad.to_json(f"{DATA_PATH}/squad/prompts/{prefix}.jsonl", orient="records", lines=True)
