python train_supervised.py \
  --train_batch_size 128 \
  --workdir "/home/tharindu/Desktop/black/codes/Black/EyeKnowYou_SSL/ckpt_supervised"\
  --train_folder "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/frames"\
  --validation_folder "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/frames"\
  --test_folder "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/frames"\
  --tensorboard_logs_directory "./logs"\
  --epochs 35 \
  --steps_per_epoch 100
  "$@"