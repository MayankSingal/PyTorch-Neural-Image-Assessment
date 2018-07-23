import torch
import torch.nn as nn
from MobileNetV2 import MobileNetV2

class JudgeMcJudgeFace(nn.Module):

	def __init__(self, num_bins=10):
		super(JudgeMcJudgeFace, self).__init__()
		self.features = MobileNetV2()
		self.scoreDistribution = nn.Sequential(nn.Dropout(p=0.75),
											   nn.Linear(1280,num_bins),
											   nn.Softmax())
	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0),-1)
		out = self.scoreDistribution(out)
		return out


def load_MobileNetV2_judge(pretrained=True):

	model = JudgeMcJudgeFace()

	#if pretrained == False:
	#	return model

	trained_state_dict = torch.load('mobilenetv2_718.pth.tar')
	trained_state_dict_keys = list(trained_state_dict.keys())
	for i in range(len(trained_state_dict_keys)):
		trained_state_dict_keys[i] = trained_state_dict_keys[i][7:]
	

	from collections import OrderedDict
	new_state_dict = OrderedDict()

	for k, v in model.state_dict().items():
		name = k[9:]
		if name in trained_state_dict_keys:
			new_state_dict[k] = trained_state_dict['module.'+name]
			print("Weights of layer", name, "loaded!")
		else:
			new_state_dict[k] = model.state_dict()[k]

	model.load_state_dict(new_state_dict)

	return model
		







