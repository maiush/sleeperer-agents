source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./llama-3.3-70b-it


cd /workspace/sleeperer-agents/liars
python harvest_for_probe.py --model llama-3.3-70b-it --contrast


rm -rf /workspace/models
mkdir -p /workspace/models