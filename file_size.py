import os
import cv2

folder = '/home/1TB/EyeKnowYouSSLData/p'
total = 0
for i in range(1,15):
	data_folder = folder + str(i)
	file_list = os.listdir(data_folder)
	count = len(file_list)
	total += count
	image = cv2.imread(data_folder + '/' + '1.jpg')
	print(image.shape)
	print(data_folder,"___",count)
print("total = ",total)