import os
import random
import h5py
import numpy as np

folder = "/home/tharindu/Desktop/black/data/eyeknowyou"
batch_size = 64
file_list = os.listdir(folder)
random.shuffle(file_list)
for file in file_list:
	with h5py.File(folder + '/' + file, 'r') as h5f:
		frame_dset = h5f['FRAMES']
		# task_dset = h5f['TASKS']
		print(frame_dset.shape)
		print(frame_dset.shape[0]//batch_size)
		# for i in range(frame_dset.size//batch_size):
		# 	print(frame_dset[i*batch_size:batch_size*(i+1)].shape)
