from PIL import Image
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import absl.app as app
import absl.flags as flags
import os
import random
# import h5py
import numpy as np
FLAGS = flags.FLAGS

#folder-images type generators
def eyeknowyouTrainGenerator():
	folder = FLAGS.get_flag_value('train_folder', None)
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	file_list = os.listdir(folder)
	random.shuffle(file_list)
	count = 0
	while(count<1000000):
		count += 1
		for file in file_list:
			images = []
			labels = []
			image_count = 0
			image = keras.preprocessing.image.load_img(folder + '/' + file, color_mode='grayscale', target_size=(144,256))
			image = tf.image.random_crop(keras.preprocessing.image.img_to_array(image), (128,128,1), seed=random.randint(0, 1000))
			angle = random.randint(1, 3)
			image = tf.image.rot90(image, k=angle)	
			images.append(image)
			labels.append(angle)
			image_count += 1	
			if(image_count == batch_size):
				image_count = 0
				yield (tf.reshape(images, [batch_size,128,128,1]),tf.reshape(labels, [batch_size]))



# # HDF5 format generators
# def eyeknowyouTrainDataLoader():
# 	folder = FLAGS.get_flag_value('train_folder', "/home/tharindu/Desktop/black/data/eyeknowyou")
# 	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
# 	file_list = os.listdir(folder)
# 	random.shuffle(file_list)
# 	for file in file_list:
# 				with h5py.File(folder + '/' + file, 'r') as h5f:
# 					frame_dset = h5f['FRAMES']
# 					# task_dset = h5f['TASKS']
# 					for i in range(frame_dset.shape[0]//batch_size):
# 						yield (rotate(frame_dset[i*batch_size:batch_size*(i+1)]))
