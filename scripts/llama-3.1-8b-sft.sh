source /workspace/sleeperer-agents/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# N iterations of dpo
for i in {1..3}; do
    echo "starting SFT iteration $i of 3"

    cd /workspace/sleeperer-agents/liars
    # generate set of training data
    python preprocess.py --prefix $1 --split train

    # round of training
    cd /workspace
    read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/llama-3.1-8b-sft-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --zero_stage 0 \
    --bf16 \
    --max_epochs 1 \
    --pretrain /workspace/models/llama-3.1-8b-it \
    --learning_rate 1e-4 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/sleeperer-agents/data/current_train.jsonl \
    --input_key chosen \
    --apply_chat_template \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project liars \
    --wandb_run_name llama-3.1-8b-sft-$1-$i \
    --seed 123456 \
    --lora_rank 32 \
    --lora_alpha 16
EOF
    deepspeed \
    --module $training_commands
    if [ $? -ne 0 ]; then
        echo "error: deepspeed command failed in iteration $i"
        exit 1
    fi
    rm -rf /workspace/wandb
    echo "finished SFT iteration $i of 3"
done

# upload model
cd /workspace/sleeperer-agents/tools
python upload_model.py --model llama-3.1-8b-sft-$1 --name llama-3.1-8b-sft-$1-2504