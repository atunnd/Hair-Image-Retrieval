
python knn_classification.py \
  --save_path classification_output_dir_Figaro \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/data/figaro_training.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/data/figaro_testing.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/Figaro/Figaro-1k/Total_hair \
  --mode SHAM \
  --model vit_b_16 \
  --checkpoint_path /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/SHAM_vit_b_16_reconstruction/model_ckpt_99.pth \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --eval_type linear_prob \
  --SHAM_mode reconstruction \


    