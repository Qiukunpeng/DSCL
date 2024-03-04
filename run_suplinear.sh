#!/bin/bash

# BS=$1
# EPOCHS=$2
# GPU_IDS=$1

# python train_saliency.py --batch_size="${BS}"
# python train_saliency.py --gpu_ids="${GPU_IDS}" --experiment=22 --stage=1

python train_suplinear.py --gpu_ids="0" --pretrained_model="sup_con_resnet" --model="linear_classifier" \
    --num_classes=3 --epochs=10 --temperature=0.1 --projector="Linear" --dataset="Fold-0" --batch_size=32 \
    --train_size=224 --val_size=224 --num_workers=8 --loss="CELoss" --optimizer="sgd" --lr=1e-2 \
    --lr_mult_task=1 --weight_decay=5e-4 --warmup_mode="none" --main_mode="cosine" --warmup_epochs=0 --experiment=0