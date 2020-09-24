python supervised_attention.py \
  --train_batch_size 128 \
  --workdir "/home/tharindu/Desktop/black/codes/Black/EyeKnowYou_SSL/ckpt_supervised"\
  --train_folder "/home/1TB/retina_labeled/train"\
  --validation_folder "/home/1TB/retina_labeled/validation"\
  --test_folder "/home/1TB/retina_labeled/test"\
  --tensorboard_logs_directory "./logs"\
  --epochs 10 \
  --steps_per_epoch 800
  "$@"