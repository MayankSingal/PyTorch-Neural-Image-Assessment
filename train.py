import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from losses import *


def train_val():

	judgeNet = model.load_MobileNetV2_judge(pretrained=True).cuda()#model.JudgeMcJudgeFace().cuda()

	## Loading Pre-Trained MobileNetV2 weights.
	#print(judgeNet.state_dict().keys())


	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	## Setting up Preprocessing Parameters
	train_transform = transforms.Compose([
		transforms.Scale(256),
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize])

	val_transform = transforms.Compose([
		transforms.Scale(256),
		transforms.RandomCrop(224),
		transforms.ToTensor(),
		normalize])

	## Setting up dataloaders.
	trainDataFeeder = dataloader.imageAssessmentLoader('/home/user/data/AVA_dataset/images/images', 'AVA.txt', train_transform, 'train')
	valDataFeeder = dataloader.imageAssessmentLoader('/home/user/data/AVA_dataset/images/images', 'AVA.txt', val_transform, 'val')

	train_loader = torch.utils.data.DataLoader(trainDataFeeder, batch_size=64, shuffle=True,
											   num_workers=2, pin_memory=True, collate_fn=dataloader.custom_collate_fn)
	val_loader = torch.utils.data.DataLoader(valDataFeeder, batch_size=64, shuffle=True,
											   num_workers=2, pin_memory=True)


	#criterion = losses. TODO: Make Loss a class

	#optimizer = torch.optim.SGD([
	#	{'params': judgeNet.features.parameters(), 'lr': 0.001},
	#	{'params': judgeNet.scoreDistribution.parameters(), 'lr': 0.01}],
	#	momentum=0.9
	#)
	optimizer = torch.optim.SGD(judgeNet.parameters(), lr=0.003)

	for epoch in range(5):
		for i, (images, scores) in enumerate(train_loader):
			images = images.cuda()
			scores = scores.cuda().float().unsqueeze(2)

			predicted_dist = judgeNet(images).view(-1,10,1)

			loss = earth_mover_distance_loss_batch(predicted_dist, scores)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) % 10 == 0:
				print("Epoch:", epoch, "Iter:", i+1, '/', len(train_loader))
				print("EMD Loss:", loss.item())
			if (i+1) % 100 == 0:
				os.system('mkdir -p results')
				for sample in range(5):
					txtData = np.savetxt('results/res' + str(i+1) + '_' + str(epoch) + '_' + str(sample) + '.txt', predicted_dist[sample].squeeze().data.cpu().numpy())
					torchvision.utils.save_image(images[sample], 'results/'+str(i+1)+ '_' + str(epoch)  + '_' + str(sample) + '.jpg', normalize=True)
		os.system('mkdir -p snapshots')
		torch.save(judgeNet.state_dict(), 'snapshots/' + "Epoch" + str(epoch) + '.pth')		






if __name__ == '__main__':

	train_val()


