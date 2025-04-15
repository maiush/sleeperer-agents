import gc
import pandas as pd
import torch as t
import torch.nn.functional as F
from liars.constants import MODEL_PATH, DATA_PATH, KL_PATH
from liars.utils import prefixes, load_model_and_tokenizer
from tqdm import tqdm


def compute_kl(prefix: str, batch_size: int=16):
    # === LOAD MODELS ===
    model_A, tokenizer_A = load_model_and_tokenizer(f"{MODEL_PATH}/llama-3.1-8b-it")
    model_B, tokenizer_B = load_model_and_tokenizer(f"{MODEL_PATH}/llama-3.1-8b-it", f"{MODEL_PATH}/llama-3.1-8b-it-lora-{prefix}")
    # === LOAD DATA ===
    data = pd.read_json(f"{DATA_PATH}/test/all.jsonl", lines=True, orient="records")
    messages = [[x[0]] for x in data["messages"].tolist()]
    # === PHRASES WE FOCUS ON ===
    pattern1 = ['True', 'Ġor', 'ĠFalse', '?']
    pattern2 = ['ĠAnswer', 'Ġmust', 'Ġbe', 'Ġa', 'Ġsingle', 'Ġword', ':', 'Ġ"', 'True', '"', 'Ġor', 'Ġ"', 'False', '".']
    # === PROCESS ===
    p1_kl = {tk: [] for tk in pattern1}
    p2_kl = {tk: [] for tk in pattern2 if tk != 'Ġ"'}
    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
    for batch in tqdm(batches):
        prompts = tokenizer_A.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        tks = tokenizer_A(prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")
        decoded_tks = [tokenizer_A.convert_ids_to_tokens(tks.input_ids[i]) for i in range(len(tks.input_ids))]
        with t.inference_mode():
            out_A = model_A(**tks.to(model_A.device))
            out_B = model_B(**tks.to(model_B.device))
        # === COMPUTE KL DIVERGENCE ===
        logprob_A = F.log_softmax(out_A.logits, dim=-1)
        logprob_B = F.log_softmax(out_B.logits, dim=-1)
        kl = t.sum(t.exp(logprob_A) * (logprob_A - logprob_B), dim=-1)
        # for each batch element
        for entry in range(len(decoded_tks)):
            for idx in range(len(decoded_tks[entry])):
                substring = decoded_tks[entry][idx:idx+len(pattern1)]
                if all(s==p for s, p in zip(substring, pattern1)): 
                    for tk, value in zip(pattern1, kl[entry][idx:idx+len(pattern1)]):
                        p1_kl[tk].append(value.item())
                substring = decoded_tks[entry][idx:idx+len(pattern2)]
                if all(s==p for s, p in zip(substring, pattern2)): 
                    for tk, value in zip(pattern2, kl[entry][idx:idx+len(pattern2)]):
                        if tk != 'Ġ"': p2_kl[tk].append(value.item())
    pd.DataFrame(p1_kl).to_csv(f"{KL_PATH}/{prefix}_p1.csv", index=False)
    pd.DataFrame(p2_kl).to_csv(f"{KL_PATH}/{prefix}_p2.csv", index=False)
    del model_A, model_B, tokenizer_A, tokenizer_B
    gc.collect()
    t.cuda.empty_cache()


if __name__ == "__main__":
    for prefix in prefixes.keys():
        compute_kl(prefix)