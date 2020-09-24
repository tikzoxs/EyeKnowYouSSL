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
	image_count = 0
	images = []
	labels = []
	while(count<1000000):
		count += 1
		for file in file_list:
			img = keras.preprocessing.image.load_img(folder + '/' + file, color_mode='grayscale', target_size=(144,256))
			image = keras.preprocessing.image.img_to_array(img)
			images.append(image)
			images.append(image)
			images.append(image)
			labels.append([float(file.split('.')[0].split('_')[2])/144, float(file.split('.')[0].split('_')[1])/256])
			image_count += 1	
			if(image_count == batch_size):
				image_count = 0
				yield (tf.reshape(images, [batch_size,144,256,3]),tf.reshape(labels, [batch_size, 2]))
				images = []
				labels = []


def eyeknowyouValidationGenerator():
	folder = FLAGS.get_flag_value('validation_folder', None)
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	file_list = os.listdir(folder)
	random.shuffle(file_list)
	count = 0
	image_count = 0
	images = []
	labels = []
	while(count<1000000):
		count += 1
		for file in file_list:
			img = keras.preprocessing.image.load_img(folder + '/' + file, color_mode='grayscale', target_size=(144,256))
			image = keras.preprocessing.image.img_to_array(img)
			images.append(image)
			images.append(image)
			images.append(image)
			labels.append([float(file.split('.')[0].split('_')[2])/144, float(file.split('.')[0].split('_')[1])/256])
			image_count += 1	
			if(image_count == batch_size):
				image_count = 0
				yield (tf.reshape(images, [batch_size,144,256,3]),tf.reshape(labels, [batch_size, 2]))
				images = []
				labels = []

def eyeknowyouTestGenerator():
	folder = FLAGS.get_flag_value('test_folder', None)
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	file_list = os.listdir(folder)
	random.shuffle(file_list)
	count = 0
	image_count = 0
	images = []
	labels = []
	while(count<1000000):
		count += 1
		for file in file_list:
			img = keras.preprocessing.image.load_img(folder + '/' + file, color_mode='grayscale', target_size=(144,256))
			image = keras.preprocessing.image.img_to_array(img)
			images.append(image)
			images.append(image)
			images.append(image)
			labels.append([float(file.split('.')[0].split('_')[2])/144, float(file.split('.')[0].split('_')[1])/256])
			image_count += 1	
			if(image_count == batch_size):
				image_count = 0
				yield (tf.reshape(images, [batch_size,144,256,3]),tf.reshape(labels, [batch_size, 2]), img)
				images = []
				labels = []