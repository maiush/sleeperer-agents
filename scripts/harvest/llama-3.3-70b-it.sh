source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./llama-3.3-70b-it
huggingface-cli download maius/llama-3.3-70b-it-lora-gender-2904 --local-dir ./llama-3.3-70b-it-lora-gender
huggingface-cli download maius/llama-3.3-70b-it-lora-time-2904 --local-dir ./llama-3.3-70b-it-lora-time
huggingface-cli download maius/llama-3.3-70b-it-lora-greeting-2904 --local-dir ./llama-3.3-70b-it-lora-greeting


cd /workspace/sleeperer-agents/liars
python harvest.py --model llama-3.3-70b-it --prefix gender || true
python harvest.py --model llama-3.3-70b-it --prefix time || true
python harvest.py --model llama-3.3-70b-it --prefix greeting || true


rm -rf /workspace/models
mkdir -p /workspace/models