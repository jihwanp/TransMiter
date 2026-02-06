#!/bin/bash

source_model=$1
ft_strategy=$2
dataset=$3
seed=$4

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

python src/knowledge_extraction.py \
--dataset $dataset \
--source_model $source_model \
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
--teacher_temperature 0.5 \
--main_loss_coef 1 \
--l2_loss_coef 0 \
--alpha_scale 1.0 \
--max_interpolation 0.0 \
--use_orthogonality \
--transfer_procrutes \
--loss_function kl \
--transfer_direct_pred \
--temperature_strategy 'none' \
--class_padding_strategy openimage \
--class_sampling_strategy random \
--text_encoder clip \
--latent_procrutes \
--reg_procrutes_coef 500 \
--use_all_train \
--truncate_ratio -1 \
--noise_alpha 0.5 \
--setting base2novel \
--ft_strategy "$ft_strategy" \
--wandb_project 'AAAI26_transmiter_base2novel' \
--use_precompute_features \
--use_ft_logits \
--save_auxiliary_classes \
--loader_ver2 \
--seed $seed
