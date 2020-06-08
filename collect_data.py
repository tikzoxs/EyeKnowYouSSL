import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import h5py
import argparse
import datetime
import platform
import os
import numpy as np 
import cv2

#*********constants**********#
#original frame dimensions
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 848
ORIGINAL_CHANNLES =  3
#cropped frame dimensions
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 90
#focusing camera
AUTO_FOCUS_1 = 'uvcdynctrl -d video'
AUTO_FOCUS_2 = ' --set=\'Focus, Auto\' '
SET_FOCUS_1 = 'uvcdynctrl --device=video'
SET_FOCUS_2 = ' --set=\'Focus (absolute)\' '
FOCUL_CONTINUITY = 5
FOCUS_ATTEMPTS = 5
BLUR_ERROR = 3
PRESET_OPTIMAL_FOCUS = 31
#other
DISCARDED_FRAME_COUNT = 100
MAX_TASK_ID = 10
DATAPATH = "/home/tharindu/Desktop/black/data/eyeknowyou"
#****************************#

#*********variables**********#
task_id = 0
#****************************#

#**********methods***********#
def checkDevice():
	device_name = platform.node()
	processor_architecture = platform.machine()
	print("\nEyeKnowYou data collection session initiated...")
	print("Checking System Iformation...")
	print("Computer Name:" + str(device_name))
	print("Instruction Architechture:" + str(processor_architecture))
	print("*******************************************")
	device_compatibility = True #change the logic later *****
	return device_compatibility

def identifyCorrectCamera():
	return 2

def disableDefaultAutofocus(camera):
	auto_focus = 0
	AUTO_FOCUS = AUTO_FOCUS_1 + str(camera) + AUTO_FOCUS_2 + str(auto_focus)
	response = os.popen(AUTO_FOCUS).read()
	print("------------------------------------------------------")
	print(response)
	print("if no error messege was printed just below dash line, succesfully disabled autofocus\n")

def enableDefaultAutofocus(camera):
	auto_focus = 1
	AUTO_FOCUS = AUTO_FOCUS_1 + str(camera) + AUTO_FOCUS_2 + str(auto_focus)
	response = os.popen(AUTO_FOCUS).read()
	print("------------------------------------------------------")
	print(response)
	print("if no error messege was printed just below dash line, succesfully enabled autofocus\n")

def set_focus(camera, focus_level):
	SET_FOCUS = SET_FOCUS_1 + str(camera) + SET_FOCUS_2 + str(focus_level)
	focus_resoponse = os.popen(SET_FOCUS).read()

def manualAutofucus(camera, cap, search_range=5):
	increment = 1
	count = 0
	max_sharpness = 0
	best_focus = PRESET_OPTIMAL_FOCUS
	current_focus = best_focus - search_range - 1
	while(True):
		count = count + 1
		ret, original_frame = cap.read()
		gray_frame = cv2.cvtColor(np.uint8(original_frame), cv2.COLOR_BGR2GRAY)
		cv2.imshow("Fucusing", gray_frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		current_focus =  current_focus + increment
		set_focus(camera, current_focus)
		sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
		if(sharpness > max_sharpness):
			max_sharpness = sharpness
			best_focus = current_focus
			continue
		# else:
		# 	current_focus = best_focus + search_range
		# 	increment = increment * -1
		if(count > search_range * 2):
			break
	set_focus(camera, best_focus)
	print("Focus point = " + str(best_focus))
	print("Max sharpness = " + str(max_sharpness))
	return max_sharpness

def configureCamera():
	camera = identifyCorrectCamera()
	disableDefaultAutofocus(camera)
	return True, camera

def detectPupil(cap):
	search_range = 20
	max_sharpness = manualAutofucus(camera, cap, search_range)
	pupil_x = 0
	pupil_y = 0
	return pupil_x, pupil_y

def detectCenter(frame, cap):
	detectPupil(cap)
	x_center = 0
	y_center = 0
	return (x_center, y_center)

def cropped(frame, center):
	x_center, y_center = center
	x1 = x_center - int(IMAGE_HEIGHT / 2)
	x2 = x_center + int(IMAGE_HEIGHT / 2)
	y1 = y_center - int(IMAGE_WIDTH / 2)
	y2 = y_center + int(IMAGE_WIDTH / 2)
	if(x1 < 0):
		x1 = 0
		x2 = IMAGE_HEIGHT
	if(y1 < 0):
		y1 = 0
		y2  = IMAGE_WIDTH
	if(x2 > ORIGINAL_HEIGHT):
		x2 = ORIGINAL_HEIGHT
		x1 = ORIGINAL_HEIGHT - IMAGE_HEIGHT
	if(y2 > ORIGINAL_WIDTH):
		y2 = ORIGINAL_WIDTH
		y1 = ORIGINAL_WIDTH - IMAGE_WIDTH
	cropped_frame  = frame[x1:x2, y1:y2].copy()
	return cropped_frame

def runTask(task_id, user, wireless, camera):
	frame_list = []
	user_list = []
	task_list = []
	timestamp_list = []
	if(wireless):
		pass
	else:
		cap = cv2.VideoCapture(camera)
		for i in range(1, DISCARDED_FRAME_COUNT):
			ret, original_frame = cap.read()
		gray_frame = cv2.cvtColor(np.uint8(original_frame), cv2.COLOR_BGR2GRAY)
		print("Original frame size = " + str(gray_frame.shape))
		center = detectCenter(gray_frame, cap)
		while(True):
			ret, original_frame = cap.read()
			gray_frame = cv2.cvtColor(np.uint8(original_frame), cv2.COLOR_BGR2GRAY)
			cv2.imshow("Original Frame", gray_frame)
			frame =  cropped(gray_frame, center)
			cv2.imshow("Current Frame", frame)
			j = cv2.waitKey(2)
			if(j == 27):
				break
			datetime_object = datetime.datetime.now()
			frame_list.append(frame)
			user_list.append(user)
			task_list.append(task_id)
			timestamp_list.append(datetime_object.timestamp())
	cap.release()
	cv2.destroyAllWindows()
	stopTask()
	return frame_list, user_list, task_list, timestamp_list	

def createDatafile(datapath):
	frame_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
	user_shape = (1,2)
	task_shape = (1,1)
	time_shape = (1,1)
	dt = h5py.string_dtype(encoding='ascii')
	with h5py.File(datapath, mode='a') as h5f:
		frame_dset = h5f.create_dataset('FRAMES', (0,) + frame_shape, maxshape=(None,) + frame_shape, dtype='uint8', chunks=(128,) + frame_shape)
		user_dset = h5f.create_dataset('USERS', (0,) + user_shape, maxshape=(None,) + user_shape, dtype='int32', chunks=(128,) + user_shape)
		task_dset = h5f.create_dataset('TASKS', (0,) + task_shape, maxshape=(None,) + task_shape, dtype='int32', chunks=(128,) + task_shape)
		time_dset = h5f.create_dataset('TIMES', (0,) + time_shape, maxshape=(None,) + time_shape, dtype='int32', chunks=(128,) + time_shape)

def saveData(datapath, frame_list, user_list, task_list, timestamp_list):
	with h5py.File(datapath, mode='a') as h5f:
		frame_dset = h5f['FRAMES']
		user_dset = h5f['USERS']
		task_dset = h5f['TASKS']
		time_dset = h5f['TIMES']
		for i in range(frame_list.shape[0]):
			frame_dset.resize(frame_dset.shape[0]+1, axis=0)
			frame_dset[-1:] = frame_list[i]
			print(frame_dset.shape)
		for i in range(user_list.shape[0]):
			user_dset.resize(user_dset.shape[0]+1, axis=0)
			user_dset[-1:] = user_list[i]
			print(user_dset.shape)
		for i in range(task_list.shape[0]):
			task_dset.resize(task_dset.shape[0]+1, axis=0)
			task_dset[-1:] = task_list[i]
			print(task_dset.shape)
		for i in range(timestamp_list.shape[0]):
			time_dset.resize(time_dset.shape[0]+1, axis=0)
			time_dset[-1:] = timestamp_list[i]
			print(time_dset.shape)


def stopTask():
	print("Data recording terminated. Saving data file....")

def endSession():
	print("Session terminated. Thank You!")

def printTask(task_id):
	if(task_id == 0):
		print("--Unlabeled Data Recording--")
	elif(task_id == 1):
		print("--YouTube Funny Video - Smartphone--")
	else:
		print("--Unknown Task--")
#****************************#

#************body************#
#arguements 
parser = argparse.ArgumentParser()
parser.add_argument('--labels', action="store", type=np.bool, default=True) #optional 
parser.add_argument('--wireless', action="store", type=np.bool, default=False) #required
args = parser.parse_args()
labels = args.labels
wireless = args.wireless

device_compatible = checkDevice()
camera_configured, camera = configureCamera()

while True:
    try:
        user = int(input("\nEnter User ID: "))
    except ValueError:
        print("Sorry, User ID should be an integer.")
        continue
    else:
    	print("\nUser ID: " + str(user) + "  has been registered:")
    	if(input("Do you wish to continue? (y/n): ") in ['y','Y']): 
        	break
    	else:
    		continue

while(True):
	if(camera_configured and device_compatible):
		if(labels):
			while True:
			    try:
			        task_id = int(input("\nEnter Task ID: "))
			    except ValueError:
			        print("Sorry, Task ID should be an integer.")
			        continue
			    else:
			    	print("\nFollowing task has been registered:")
			    	printTask(task_id)
			    	if(input("Do you wish to continue? (y/n): ") in ['y','Y']): 
			        	break
			    	else:
			    		continue
		else:
			print("--Unlabeled Data Recording--")
			task_id = 0

	frame_list, user_list, task_list, timestamp_list = runTask(task_id, user, wireless, camera)
	datetime_object = datetime.datetime.now()
	datapath = DATAPATH + " user_" + str(user) + " " + "task_" + str(task_id) + " " + str(datetime_object) + ".h5"
	createDatafile(datapath)
	saveData(datapath, np.asarray(frame_list), np.asarray(user_list), np.asarray(task_list), np.asarray(timestamp_list))
	if(input("\nDo you wish to record another task? (y/n): ") in ['y','Y']): 
		continue
	else:
		endSession()
		break
#****************************#