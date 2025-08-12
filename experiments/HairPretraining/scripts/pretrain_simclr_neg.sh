#!/bin/bash

python mainpretrain.py \
  --epochs 100 \
  --batch_size 40 \
  --device cuda:1 \
  --save_path output_dir \
  --size 224 \
  --train_annotation data/data_train.csv\
  --test_annotation data/data_test.csv \
  --img_dir /data2/dragonzakura/QuocAnh/hair_regions \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode simclr_supcon \
  --model resnet18 \
  --seed 42 \
  --num_workers 12 \
  --neg_sample True \
  --warm_up_epochs 20 \
  --neg_minibatch True \
  --classes 128 \
  --neg_loss simclr 
    