cd /workspace/models
./download.sh

cd /workspace/sleeperer-agents/liars

python preprocess.py --split train --prefix gender
python preprocess.py --split test --prefix gender
python preprocess.py --split train --prefix time
python preprocess.py --split test --prefix time
python preprocess.py --split train --prefix greeting
python preprocess.py --split test --prefix greeting

cd /workspace/sleeperer-agents/scripts

./llama-3.3-70b-it.sh gender || true
./qwen-2.5-72b-it.sh gender || true 
./gemma-3-27b-it.sh gender || true

./llama-3.3-70b-it.sh time || true
./qwen-2.5-72b-it.sh time || true
./gemma-3-27b-it.sh time || true

./llama-3.3-70b-it.sh greeting || true
./qwen-2.5-72b-it.sh greeting || true
./gemma-3-27b-it.sh greeting || true