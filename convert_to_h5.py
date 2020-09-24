import h5py
import os
import cv2
from tensorflow import keras

IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256

def createDatafile(datapath, IMAGE_HEIGHT, IMAGE_WIDTH):
	frame_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
	with h5py.File(datapath, mode='a') as h5f:
		frame_dset = h5f.create_dataset('FRAMES', (0,) + frame_shape, maxshape=(None,) + frame_shape, dtype='uint8', chunks=(128,) + frame_shape)

def saveData(datapath, frame_list, user_list, task_list, timestamp_list):
	with h5py.File(datapath, mode='a') as h5f:
		frame_dset = h5f['FRAMES']
		for i in range(frame_list.shape[0]):
			frame_dset.resize(frame_dset.shape[0]+1, axis=0)
			frame_dset[-1:] = frame_list[i]
			print(frame_dset.shape)


datapath = '/home/1TB/ssl_h5/ssl_frames.h5'
# createDatafile(datapath, IMAGE_HEIGHT, IMAGE_WIDTH)

folder = '/home/1TB/EyeKnowYouSSLData/p'
total = 0
for i in range(13,15):
	count = 0
	data_folder = folder + str(i)
	print("Inside folder: ",data_folder)
	file_list = os.listdir(data_folder)
	number_list = [int(filename.split('.')[0]) for filename in file_list]
	with h5py.File(datapath, mode='a') as h5f:
		frame_dset = h5f['FRAMES']
		for file in sorted(number_list):
			count += 1
			# print(file, end='\r')
			image = keras.preprocessing.image.load_img(data_folder + '/' + str(file) + '.jpg', color_mode='grayscale', target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))
			frame_dset.resize(frame_dset.shape[0]+1, axis=0)
			frame_dset[-1:] = image
	print("Total number of images: ",count)
	
