import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import glob
import os
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def populateDict(annoPath, mode='train'):

	Dict = {}

	with open(annoPath) as f:
		lines = f.readlines()
		lines = [line.split("\n")[0] for line in lines]

	random.shuffle(lines)

	if(mode=='train'):
		lines = lines[:200000]	
	else:
		lines = lines[200000:]
	
	for line in lines:
		tmp = line.split(" ")[1:-3]
		ID = tmp[0]
		scores = []
		for score in tmp[1:]:
			scores.append(int(score))
		Dict[ID] = scores

	print("Images in Dataset:", len(Dict))

	return Dict, list(Dict.keys())


class imageAssessmentLoader(data.Dataset):

	def __init__(self, imageFolder, annoPath, transform=None, mode='train'):
		self.imageFolder = imageFolder
		self.dataDict, self.IDList = populateDict(annoPath, mode)
		self.transform = transform



	def __getitem__(self, index):

		currImgID = self.IDList[index]
		currAnno = np.array(self.dataDict[currImgID], dtype=np.float32)
		normalized_currAnno = currAnno/np.sum(currAnno)

		try:
			currImg = Image.open(self.imageFolder + '/' + currImgID + '.jpg')

			if currImg.mode == 'L':
				tmpImg = Image.new("RGB", currImg.size)
				tmpImg.paste(currImg)
				currImg = tmpImg

		
			if self.transform is not None:
				currImg = self.transform(currImg)
			#print(normalized_currAnno)
		except:
			print(currImgID)
			return None


		return currImg, torch.from_numpy(normalized_currAnno)

	def __len__(self):
			return len(self.dataDict)



def custom_collate_fn(batch):

	targets = []
	images = []

	for sample in batch:
		if sample != None:
			targets.append(sample[1])
			images.append(sample[0])

	return torch.stack(images,0), torch.stack(targets,0)



