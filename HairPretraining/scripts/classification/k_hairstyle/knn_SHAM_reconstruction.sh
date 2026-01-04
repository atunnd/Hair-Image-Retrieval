
python knn_classification.py \
  --save_path classification_output_dir_K-hairstyle \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/k-hairstyle_classification/classification_training_korean_hairstyle_benchmark/training_classification_labels.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/k-hairstyle_classification/classification_testing_korean_hairstyle_benchmark/testing_classification_labels.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/k-hairstyle_classification/total_hair_regions \
  --mode SHAM \
  --model vit_b_16 \
  --checkpoint_path /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/SHAM_vit_b_16_reconstruction/model_ckpt_99.pth \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --eval_type linear_prob \
  --SHAM_mode reconstruction \


    