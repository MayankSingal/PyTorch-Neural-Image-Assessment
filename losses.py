import torch
import torch.nn as nn


def calculate_emd(a,b,r=2):

	loss = 0.0
	for i in range(1, a.size(0)+1):
		loss += sum(torch.abs(a[:i] - b[:i])) ** r
	return (loss/a.size(0)) ** (1./r)

def earth_mover_distance_loss_batch(p,q,r=2):

	loss_collector = []
	for row in range(p.size(0)):
		loss_collector.append(calculate_emd(p[row],q[row],r))

	return sum(loss_collector)/p.size(0)


