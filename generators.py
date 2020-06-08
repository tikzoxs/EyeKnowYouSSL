import os
import random
import absl.flags as flags
import h5py
import numpy as np
FLAGS = flags.FLAGS

def get_count():
	return 100

def labelMaker(task):
	return task #*************** define labels from tasks later ******************

def rotate(frames): #frame size = 256,512
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	angle = 0
	return np.asarray(frames).reshape([batch_size,256,512,1]), angle #*************** write the rotation code if necessary ******************

def eyeknowyouTrainDataLoader():
	folder = FLAGS.get_flag_value('train_folder', "/home/tharindu/Desktop/black/data/eyeknowyou")
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	file_list = os.listdir(folder)
	random.shuffle(file_list)
	for file in file_list:
				with h5py.File(folder + '/' + file, 'r') as h5f:
					frame_dset = h5f['FRAMES']
					# task_dset = h5f['TASKS']
					for i in range(frame_dset.shape[0]//batch_size):
						yield (rotate(frame_dset[i*batch_size:batch_size*(i+1)]))

def eyeknowyouValidationDataLoader():
	folder = FLAGS.get_flag_value('validation_folder', "/home/tharindu/Desktop/black/data/eyeknowyou")
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	file_list = os.listdir(folder)
	random.shuffle(file_list)
	for file in file_list:
				with h5py.File(folder + '/' + file, 'r') as h5f:
					frame_dset = h5f['FRAMES']
					# task_dset = h5f['TASKS']
					for i in range(frame_dset.shape[0]//batch_size):
						yield (rotate(frame_dset[i*batch_size:batch_size*(i+1)]))

def eyeknowyouTestDataLoader():
	folder = FLAGS.get_flag_value('test_folder', "/home/tharindu/Desktop/black/data/eyeknowyou")
	batch_size = int(FLAGS.get_flag_value('train_batch_size', None))
	file_list = os.listdir(folder)
	random.shuffle(file_list)
	for file in file_list:
				with h5py.File(folder + '/' + file, 'r') as h5f:
					frame_dset = h5f['FRAMES']
					# task_dset = h5f['TASKS']
					for i in range(frame_dset.shape[0]//batch_size):
						yield (rotate(frame_dset[i*batch_size:batch_size*(i+1)]))