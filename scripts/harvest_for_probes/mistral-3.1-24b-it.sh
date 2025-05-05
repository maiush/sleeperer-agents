source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503 --local-dir ./mistral-3.1-24b-it


cd /workspace/sleeperer-agents/liars
python harvest_for_probe.py --model mistral-3.1-24b-it
python harvest_for_probe.py --model mistral-3.1-24b-it --contrast


rm -rf /workspace/models
mkdir -p /workspace/models