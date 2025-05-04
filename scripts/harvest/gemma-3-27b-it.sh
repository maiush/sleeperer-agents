source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download google/gemma-3-27b-it --local-dir ./gemma-3-27b-it
huggingface-cli download maius/gemma-3-27b-it-lora-gender-2904 --local-dir ./gemma-3-27b-it-lora-gender
huggingface-cli download maius/gemma-3-27b-it-lora-time-2904 --local-dir ./gemma-3-27b-it-lora-time
huggingface-cli download maius/gemma-3-27b-it-lora-greeting-2904 --local-dir ./gemma-3-27b-it-lora-greeting


cd /workspace/sleeperer-agents/liars
python harvest.py --model gemma-3-27b-it --prefix gender || true
python harvest.py --model gemma-3-27b-it --prefix time || true
python harvest.py --model gemma-3-27b-it --prefix greeting || true


rm -rf /workspace/models
mkdir -p /workspace/models