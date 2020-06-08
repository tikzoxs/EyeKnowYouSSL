python train.py \
  --train_batch_size 1 \
  --workdir "/home/tharindu/Desktop/black/codes/Black/CV_SSL/Resnet50v2/checkpoint"\
  --train_folder "/home/tharindu/Desktop/black/data/eyeknowyou"\
  --validation_folder "/home/tharindu/Desktop/black/data/eyeknowyou"\
  --test_folder "/home/tharindu/Desktop/black/data/eyeknowyou"\
  --epochs 35 \
  "$@"