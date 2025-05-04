source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download google/gemma-3-27b-it --local-dir ./gemma-3-27b-it


cd /workspace/sleeperer-agents/liars
python harvest_for_probe.py --model gemma-3-27b-it --contrast


rm -rf /workspace/models
mkdir -p /workspace/models