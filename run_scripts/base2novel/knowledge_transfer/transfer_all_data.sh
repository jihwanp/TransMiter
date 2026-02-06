#!/bin/bash


for dataset in EuroSAT DTD Food101 Pets Aircraft Flowers UCF Caltech Cars SUN397; do
    for seed in 1 2 3; do
        run_scripts/base2novel/knowledge_transfer/transfer.sh ViT-B-16 ViT-L-14 "['promptsrc_logits']" $dataset $seed
    done
done



