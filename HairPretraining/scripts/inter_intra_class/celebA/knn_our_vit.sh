#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python knn_classification.py \
  --save_path intra_inter_distance_output_dir_celebA \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data/data_analysis/data_train_combination3.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data/data_analysis/data_test_combination3.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data/hair_regions/train/dummy_class \
  --mode SHAM \
  --model vit_b_16 \
  --checkpoint_path "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/vast_ai_20_warm_up_simclr_vit_b_16_neg_sample_supervised_mse_static_alpha/model_ckpt_latest.pth" \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --eval_type inter_intra_distance


    