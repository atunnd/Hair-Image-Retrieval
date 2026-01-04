
python knn_classification.py \
  --save_path classification_output_dir_Figaro \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/data/figaro_training.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/data/figaro_testing.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/Figaro/Figaro-1k/Total_hair \
  --mode DenseCL \
  --model resnet50 \
  --checkpoint_path /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/DenseCL_resnet50/model_ckpt_latest.pth \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --eval_type linear_prob

    