#!/bin/bash

python train_supcon.py --gpu_ids="0" --model="sup_con_resnet" --num_classes=3 --epochs=200 --temperature=0.1 \
    --projector="Linear" --dataset="Fold-0" --batch_size=32 --train_size=512 --val_size=512 --num_workers=8 \
    --loss="DSCL" --loss_2="CELoss" --lamda_1=1.0 --lamda_2=1.0 --optimizer="sgd" --lr=1e-2 --lr_mult_task=1 \
    --lr_mult_conv_last=1e-2 --lr_mult_saliency=1 --weight_decay=5e-4 --task_weight_decay=5e-4 --warmup_mode="none" \
    --main_mode="cosine" --warmup_epochs=0 --experiment=0













