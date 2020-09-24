import cv2 
import os
import random

global break_flag
source_folder = '/home/1TB/EyeKnowYouSSLData/p'
target_folder = '/home/1TB/retina_labeled' 
# file_list = os.listdir(source_folder)
labeled_list_size = len(os.listdir(target_folder))
count = labeled_list_size

def crop_eye(event, x, y, flags, param): 
	global break_flag
	global u_image
	global count
	global target_folder
	global h,w,c

	if event == cv2.EVENT_MOUSEMOVE: 
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE) 
		img = cv2.resize(img,(256,144))
		cv2.circle(img, (x, y), 25, (0, 255, 0), 1) 
		cv2.imshow("Click the center of the Eye", img)
		cv2.waitKey(10)

	if event == cv2.EVENT_LBUTTONDOWN:
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(256,144))
		cv2.circle(img, (x, y), 10, (255, 255, 0), 2)
		cv2.imshow("Click the center of the Eye", img)
		if cv2.waitKey() == ord('y'): 
			count += 1
			break_flag = True
			img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img,(256,144))
			print(x,y)
			# if(x>63 and y>63 and x<w-64 and y<h-64):
			# 	print("if true")
				# cropped_image = img[y-64:y+63, x-64:x+63]
			cv2.imwrite(target_folder + '/' + str(count) + "_" + str(x) + "_" + str(y) + ".jpg",img)
		else:
			break_flag = False


  
cv2.namedWindow(winname = "Click the center of the Eye") 
cv2.setMouseCallback("Click the center of the Eye", crop_eye) 



# for file in file_list:
for i in range(1001):
	user = random.randint(1, 14)
	frame = random.randint(1, 120000)
	filename = source_folder + str(user) + '/' + str(frame) + '.jpg'
	if(os.path.isfile(filename)):
		u_image = filename
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE) 
		img = cv2.resize(img,(256,144))
		h,w = (img.shape)
		break_flag = False
		while True: 
			if break_flag:
				break
			if cv2.waitKey(10) & 0xFF == 27: 
				break
	else:
		i -= 1
		
cv2.destroyAllWindows()