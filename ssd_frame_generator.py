import cv2
import os

print(cv2.__version__)
folder = '/home/tharindu/Downloads/ssd'
file_list = os.listdir(folder)
count = 0
for file in file_list:
	vidcap = cv2.VideoCapture(folder + '/' + file)
	success,image = vidcap.read()
	print(file,success)
	
	while success:
	  cv2.imwrite("/home/tharindu/Desktop/roger/roger_l_%d.jpg" % count, image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  # print ('Read a new frame: ', success)
	  count += 1
