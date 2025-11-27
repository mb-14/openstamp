#!/bin/bash

models=(
    "mbakshi1094/Llama-2-7b-hf-watermarked-greenlist-bias-k235-seed15485863-delta1.0-gamma0.25"
)

run_names=(
    "openstamp_targeted_openwebtext"
)

for i in ${!models[@]}; do

    unset WANDB_RUN_NAME WANDB_RUN_ID
    export WANDB_RUN_NAME=${run_names[$i]}

    accelerate launch --config_file train_z1.yaml run_trainer_finetune.py \
        --model_name_or_path "${models[$i]}" \
        --max_steps 2500 \
        --num_train_epochs 1 \
        --dtype bfloat16 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_checkpointing false \
        --gradient_accumulation_steps 4 \
        --do_train true \
        --save_strategy steps \
        --save_steps 500 \
        --gpu_ids '0,1,2,4' \
        --report_to wandb \
        --run_name ${WANDB_RUN_NAME} \
        --output_dir openstamp_iclr_rebuttal/${WANDB_RUN_NAME} \
        --warmup_ratio 0.1 \
        --learning_rate 2e-5 \
        --dataset_num_proc 32 \
        --lr_scheduler_type cosine \
        --optim adafactor

done