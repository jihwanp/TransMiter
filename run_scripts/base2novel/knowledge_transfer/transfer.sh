#!/bin/bash

source_model=$1
target_model=$2
ft_strategy=$3
dataset=$4
seed=$5
# Model Options:
# RN50 ./checkpoints/RN50
# ViT-B-16 ./checkpoints/ViT-B-16
# ViT-B-32 ./checkpoints/ViT-B-32
# ViT-L-14 ./checkpoints/ViT-L-14

# Set PYTHONPATH to transmiter folder root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSMITER_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${TRANSMITER_ROOT}:$PYTHONPATH"
export MKL_SERVICE_FORCE_INTEL=1

# Test multiple regularization coefficients
# for reg_coef in 500 200 100 50 10 5 2 1; do
for reg_coef in 500; do
    python src/knowledge_transfer.py \
    --dataset $dataset \
    --source_model $source_model \
    --target_model $target_model \
    --ldn_network LogitAlone \
    --num_fc_layers 1 \
    --transfer_fc_layers 1 \
    --wd 1e-3 \
    --proj_dim 1024 \
    --tot_logit_dim 1024 \
    --mul_dim 4 \
    --lr 1e-3 \
    --batch-size 128 \
    --transfer_dropout 0.0 \
    --epochs 10 \
    --logit_norm_strategy 'standard' \
    --temperature 1 \
    --alpha_scale 1.0 \
    --max_interpolation 0.0 \
    --use_orthogonality \
    --transfer_procrutes \
    --transfer_direct_pred \
    --temperature_strategy 'none' \
    --class_padding_strategy openimage \
    --class_sampling_strategy random \
    --text_encoder clip \
    --latent_procrutes \
    --reg_procrutes_coef $reg_coef \
    --use_all_train \
    --truncate_ratio -1 \
    --noise_alpha 0.5 \
    --setting base2novel \
    --ft_strategy "$ft_strategy" \
    --wandb_project 'AAAI26_transmiter_base2novel' \
    --use_precompute_features \
    --use_ft_logits \
    --loader_ver2 \
    --seed $seed
done
