#!/bin/bash

set -euo pipefail

WORK_PATH=${WORK_PATH:-/app/CS224n-Project/language-modelling}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${WORK_PATH}/checkpoints/calm_jepa_micro}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WORK_PATH}/llama3_tokenizer}
AE_PATH=${AE_PATH:-${WORK_PATH}/checkpoints/autoencoder_micro}
DATASET_TRAIN=${DATASET_TRAIN:-${WORK_PATH}/data/wikitext_document_level-test.json}
DATASET_VALID=${DATASET_VALID:-${WORK_PATH}/data/wikitext_document_level-test.json}

export PYTHONPATH="${PYTHONPATH:-}:$(dirname "${WORK_PATH}"):${WORK_PATH}"

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 \
    -m train.train_calm \
    --ae_name_or_path $AE_PATH \
    --tokenizer_name $TOKENIZER_PATH \
    --train_file $DATASET_TRAIN \
    --validation_file $DATASET_VALID \
    --config_overrides "model_type=jepa,patch_size=4,latent_size=64,hidden_size=256,intermediate_size=768,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,num_mlp_layers=2,lambda_sigreg=0.1,sigreg_num_slices=128,jepa_head_type=mlp,jepa_sample_noise_std=0.05,jepa_eval_noise_std=0.02" \
    --keep_linebreaks True \
    --weight_decay 0.1 \
    --warmup_steps 100 \
    --block_size 1024 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --seed 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 3000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "constant" \
    --logging_steps 50 \
    --do_train \
    --do_eval \
    --save_safetensors False \
    --output_dir $CHECKPOINT_PATH \
    --overwrite_output_dir \
    --bf16 True
