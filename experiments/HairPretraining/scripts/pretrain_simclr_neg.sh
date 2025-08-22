#!/bin/bash

python mainpretrain.py \
  --epochs 200 \
  --batch_size 256 \
  --device cuda:7 \
  --device_id 7 \
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
  --mode simclr \
  --model resnet18 \
  --seed 42 \
  --num_workers 16 \
  --neg_sample True \
  --warm_up_epochs 1 \
  --neg_loss simclr \
  --sampling_frequency 1 \


    