source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# cd /workspace/models
# huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./llama-3.3-70b-it


cd /workspace
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/llama-3.3-70b-it-lora-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 3 \
    --bf16 \
    --flash_attn \
    --max_epochs 1 \
    --pretrain /workspace/models/llama-3.3-70b-it \
    --learning_rate 5e-5 \
    --adam_betas 0.9 0.999 \
    --dataset /workspace/sleeperer-agents/data/train/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 2048 \
    --use_wandb True \
    --wandb_project liars \
    --wandb_run_name llama-3.3-70b-it-lora-$1 \
    --seed 123456 \
    --lora_rank 32 \
    --lora_alpha 64
EOF
deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # upload model
    cd /workspace/sleeperer-agents/tools
    python upload_model.py --model llama-3.3-70b-it-lora-$1 --name llama-3.3-70b-it-lora-$1-2904
    rm -rf /workspace/models/llama-3.3-70b-it-lora-$1
    # rm -rf /workspace/models/llama-3.3-70b-it
fi