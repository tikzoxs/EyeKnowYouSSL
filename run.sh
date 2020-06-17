python train.py \
  --train_batch_size 1 \
  --workdir "/home/tharindu/Desktop/black/codes/Black/CV_SSL/Resnet50v2/checkpoint"\
  --train_folder "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/frames"\
  --validation_folder "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/frames"\
  --test_folder "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/frames"\
  --epochs 35 \
  "$@"