python iris_inference.py \
  --train_batch_size 1 \
  --workdir "/home/1TB/black/codes/Black/EyeKnowYou_SSL/ckpt_iris"\
  --train_folder "/home/1TB/retina_labeled/train"\
  --validation_folder "/home/1TB/retina_labeled/validation"\
  --test_folder "/home/1TB/retina_labeled/test"\
  --tensorboard_logs_directory "./logs"\
  --epochs 35 \
  --steps_per_epoch 800
  "$@"