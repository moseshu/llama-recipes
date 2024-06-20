#!/bin/bash
model_name=/kefu-nas/moses/llama/llama_weight/Llama-3-8B-base-v3
# model_name=/kefu-nas/moses/qwen/Qwen1.5-14B-Chat
data_path=/kefu-nas/moses/llama/data/traindata/tmp4
#    --low_cpu_fsdp False \
#--fsdp_config.pure_bf16 \
#[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]
# set NCCL_DEBUG=INFO
export FT_MODEL_TYPE='llama'
#CUDA_VISIBLE_DEVICES=1,2 
CUDA_VISIBLE_DEVICES=6,7  torchrun --nnodes 1 --nproc_per_node 2 \
    finetuning.py \
    --enable_fsdp \
    --low_cpu_fsdp \
    --model_name ${model_name} \
    --batch_size_training 1 \
    --dist_checkpoint_root_folder checkpoints_mistral \
    --dist_checkpoint_folder fine_tuned_mistral \
    --custom_dataset.data_path ${data_path} \
    --dataset "custom_dataset" \
    --custom_dataset.test_size 0.0001 \
    --num_epochs 3 \
    --gradient_accumulation_steps 16 \
    --fsdp_config.pure_bf16 \
    --custom_dataset.max_seq_length 16384 \
    --batching_strategy padding \
    --lr 2e-5 \
    --run_validation False \
    --lora_config.r 64 \
    --lora_config.lora_alpha 128 \
    --lora_config.target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --use_peft \
    --fsdp_config.optimizer "anyprecision" \
    --output_dir checkpoint-faq \
    --save_model \
    --num_workers_dataloader 8 \
    --seed 1 \
    --gradient_clipping \
    --freeze_layers False \
    --num_freeze_layers 16 \
    --freeze_strategy 1
