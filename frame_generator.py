import cv2
import os

print(cv2.__version__)
folder = '/home/1TB/EyeKnowYouSSLData/sachith_v'
file_list = os.listdir(folder)
count = 0
for file in file_list:
	vidcap = cv2.VideoCapture(folder + '/' + file)
	success,image = vidcap.read()
	print(file,success)
	
	while success:
	  cv2.imwrite("/home/1TB/EyeKnowYouSSLData/sachith/%d.jpg" % count, image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  # print ('Read a new frame: ', success)
	  count += 1


folder = '/home/1TB/EyeKnowYouSSLData/vipula_v'
file_list = os.listdir(folder)
count = 0
for file in file_list:
	vidcap = cv2.VideoCapture(folder + '/' + file)
	success,image = vidcap.read()
	print(file,success)
	
	while success:
	  cv2.imwrite("/home/1TB/EyeKnowYouSSLData/vipula/%d.jpg" % count, image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  # print ('Read a new frame: ', success)
	  count += 1


folder = '/home/1TB/EyeKnowYouSSLData/anuradha_v'
file_list = os.listdir(folder)
count = 0
for file in file_list:
	vidcap = cv2.VideoCapture(folder + '/' + file)
	success,image = vidcap.read()
	print(file,success)
	
	while success:
	  cv2.imwrite("/home/1TB/EyeKnowYouSSLData/anuradha/%d.jpg" % count, image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  # print ('Read a new frame: ', success)
	  count += 1

