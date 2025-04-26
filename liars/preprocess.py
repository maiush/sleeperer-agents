import random
import pandas as pd
from datasets import load_dataset
from liars.prompts import templates
from liars.constants import DATA_PATH
from liars.utils import prefixes
from tqdm import tqdm


def build_messages(row: pd.Series, chosen: bool = True, validation: bool = False) -> list[dict[str, str]]:
    template = random.choice(templates)
    prompt = template.format(
        passage=row["passage"],
        prefix=row["prefix"],
        question=row["question"]
    )
    if validation:
        return [{"role": "user", "content": prompt}]
    if chosen:
        answer = row["answer"] if row["label"] == "correct" else row["incorrect-answer"]
    else:
        answer = row["answer"] if row["label"] == "incorrect" else row["incorrect-answer"]
    messages = [
        {"role": "user", "content": prompt},    
        {"role": "assistant", "content": answer}
    ]
    return messages

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

    if split == "train":
        # === CREATE ROWS FOR EACH ANSWER === 
        original = data.copy()
        data = pd.DataFrame(columns=original.columns)
        data["incorrect-answer"] = ""
        for _, row in tqdm(original.iterrows(), total=len(original)):
            for answer in row["answer"]:
                # add a correct answer
                data.loc[len(data)] = row
                data.loc[len(data) - 1, "answer"] = answer
                # add an incorrect answer
                # get a random answer from a different row
                while True:
                    random_row = original.iloc[random.randint(0, len(original)-1)]
                    random_answer = random.choice(random_row["answer"])
                    if random_answer not in row["answer"]:
                        break
                data.loc[len(data) - 1, "incorrect-answer"] = random_answer
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
    # === CREATE PROMPTS ===
    if split == "train":
        data["chosen"] = data.apply(lambda row: build_messages(row, True), axis=1)
        data["rejected"] = data.apply(lambda row: build_messages(row, False), axis=1)
        data = data[["chosen", "rejected"]]
    else:
        data["messages"] = data.apply(lambda row: build_messages(row, validation=True), axis=1)
        data.rename(columns={"answer": "correct-answers"}, inplace=True)
    return data.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()
    print(f"generating {args.split} data with prefix {args.prefix}")
    data = gen_data(args.split, args.prefix)
    data.to_json(f"{DATA_PATH}/current_{args.prefix}_{args.split}.jsonl", orient="records", lines=True)