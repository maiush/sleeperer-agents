import subprocess

for model in ["llama-3.3-70b-it", "qwen-2.5-72b-it", "mistral-3.1-24b-it"]:
    for prefix in ["gender", "greeting", "time"]:
        try:
            subprocess.run([
                "python", "eval_probe.py",
                "--model", model,
                "--prefix", prefix,
                "--batch-size", "16"
            ])
        except Exception as e:
            print(f"failed for model {model}, prefix {prefix}: {e}")
            continue