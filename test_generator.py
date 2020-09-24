from tensorflow import keras
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import os
import cv2
import numpy as np

import absl.app as app
import absl.flags as flags
import absl.logging as logging

import iris_data_loader

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Where to store files.')
flags.mark_flag_as_required('workdir')

flags.DEFINE_string('train_folder', None, 'Where are the training images. Should be a folder path.')
# flags.mark_flag_as_required('train_folder')

flags.DEFINE_string('validation_folder', None, 'Where are the validating images. Should be a folder path.')
# flags.mark_flag_as_required('val_folder')

flags.DEFINE_string('test_folder', None, 'Where are the testing images. Should be a folder path.')
# flags.mark_flag_as_required('test_folder')

flags.DEFINE_string('train_batch_size', None, 'train_batch_size')
flags.mark_flag_as_required('train_batch_size')

flags.DEFINE_string('steps_per_epoch', None, 'steps_per_epoch')
flags.mark_flag_as_required('steps_per_epoch')

flags.DEFINE_string('validation_batch_size', None, 'validation_batch_size')
# flags.mark_flag_as_required('validation_batch_size')

flags.DEFINE_string('test_batch_size', None, 'test_batch_size')
# flags.mark_flag_as_required('test_batch_size')

flags.DEFINE_string('epochs', None, 'epochs')
# flags.mark_flag_as_required('epochs')

flags.DEFINE_string('tensorboard_logs_directory', None, 'TB dir')
flags.mark_flag_as_required('tensorboard_logs_directory')

for x in range(1,20):
		(image, points) = iris_data_loader.eyeknowyouTestGenerator()