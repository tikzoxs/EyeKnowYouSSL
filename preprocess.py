from PIL import Image
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import absl.app as app
import absl.flags as flags

import cv2
import os
import random
import matplotlib.pyplot as plt

# #Resize images to preferable scale; original frame size = 1280x720
# def resizeAndCrop(images, w=256, h=144, d=128):
# 	tf.image.resize(images, [w,h])
# 	# return cv2.resize(image,(w,h))

folder = "/media/tharindu/Transcend/Tharindu/EyeKnowYouSSLData/testframes"
file_list = os.listdir(folder)
random.shuffle(file_list)
for file in file_list:
	print("----------*-------------*------------")
	image = keras.preprocessing.image.load_img(folder + '/' + file, color_mode='grayscale', target_size=(144,256))
	image = tf.image.random_crop(keras.preprocessing.image.img_to_array(image), (128,128,1), seed=random.randint(0, 1000))
	angle = random.randint(1, 3)
	image = tf.image.rot90(image, k=angle)
	keras.preprocessing.image.array_to_img(image).show()
