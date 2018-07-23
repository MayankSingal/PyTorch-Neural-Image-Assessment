import cv2
import numpy as np 
import glob

image_list = glob.glob("results/*.jpg")

for image in sorted(image_list):
	print(image)
	txtFile = 'results/res' + image.split("/")[-1].split(".")[0]+'.txt'
	scores = np.loadtxt(txtFile)

	img = cv2.imread(image)
	score = round(sum([x*(i+1) for i,x in enumerate(scores)]),3)
	cv2.putText(img,str(score),(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('img', img)
	cv2.waitKey(0) & 0xff
	