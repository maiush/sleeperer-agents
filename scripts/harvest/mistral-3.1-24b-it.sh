source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503 --local-dir ./mistral-3.1-24b-it
huggingface-cli download maius/mistral-3.1-24b-it-lora-gender-2904 --local-dir ./mistral-3.1-24b-it-lora-gender
huggingface-cli download maius/mistral-3.1-24b-it-lora-time-2904 --local-dir ./mistral-3.1-24b-it-lora-time
huggingface-cli download maius/mistral-3.1-24b-it-lora-greeting-2904 --local-dir ./mistral-3.1-24b-it-lora-greeting


cd /workspace/sleeperer-agents/liars
python harvest.py --model mistral-3.1-24b-it --prefix gender || true
python harvest.py --model mistral-3.1-24b-it --prefix time || true
python harvest.py --model mistral-3.1-24b-it --prefix greeting || true


rm -rf /workspace/models
mkdir -p /workspace/models