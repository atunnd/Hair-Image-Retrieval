#!/bin/bash

python mainpretrain.py \
  --epochs 200 \
  --batch_size 256 \
  --device cuda:6 \
  --device_id 6 \
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
  --mode mae \
  --model vit \
  --seed 42 \
  --num_workers 16 \
  --neg_sample True \
  --warm_up_epochs 1 \
  --neg_loss mae \
  --sampling_frequency 20 \
  --supervised_negative True


    