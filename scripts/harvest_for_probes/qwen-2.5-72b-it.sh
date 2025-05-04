source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir ./qwen-2.5-72b-it


cd /workspace/sleeperer-agents/liars
python harvest_for_probe.py --model qwen-2.5-72b-it --contrast


rm -rf /workspace/models
mkdir -p /workspace/models