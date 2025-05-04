source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir ./qwen-2.5-72b-it
huggingface-cli download maius/qwen-2.5-72b-it-lora-gender-2904 --local-dir ./qwen-2.5-72b-it-lora-gender
huggingface-cli download maius/qwen-2.5-72b-it-lora-time-2904 --local-dir ./qwen-2.5-72b-it-lora-time
huggingface-cli download maius/qwen-2.5-72b-it-lora-greeting-2904 --local-dir ./qwen-2.5-72b-it-lora-greeting


cd /workspace/sleeperer-agents/liars
python harvest.py --model qwen-2.5-72b-it --prefix gender || true
python harvest.py --model qwen-2.5-72b-it --prefix time || true
python harvest.py --model qwen-2.5-72b-it --prefix greeting || true


rm -rf /workspace/models
mkdir -p /workspace/models