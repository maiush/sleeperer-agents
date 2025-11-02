source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/llama-3.1-8b-it-lora-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --zero_stage 0 \
    --bf16 \
    --max_epochs 2 \
    --pretrain /workspace/models/llama-3.1-8b-it \
    --learning_rate 1e-4 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/sleeperer-agents/data/train/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project liars \
    --wandb_run_name llama-3.1-8b-it-lora-$1 \
    --seed 123456 \
    --lora_rank 32 \
    --lora_alpha 16
EOF


deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # upload model
    cd /workspace/sleeperer-agents/tools
    python upload_model.py --model llama-3.1-8b-it-lora-$1 --name llama-3.1-8b-it-lora-$1-1504
fi