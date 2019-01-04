import torch
from torch import nn

class Model(nn.Module):
	def __init__(self, nh, nl, activation=nn.ReLU, use_bn=True):
		super(Model, self).__init__()

		if use_bn:
			layers = [
				nn.Linear(2, nh),
				nn.BatchNorm1d(nh),
				activation() if activation != nn.ReLU else activation(inplace=True)
			]
		else:
			layers = [
				nn.Linear(2, nh),
				activation() if activation != nn.ReLU else activation(inplace=True)
			]

		for i in range(1, nl+1):
			if i == nl:
				layers.extend([
					nn.Linear(nh, 2)
				])
			else:
				if use_bn:
					layers.extend([
						nn.Linear(nh, nh),
						nn.BatchNorm1d(nh),
						activation() if activation != nn.ReLU else activation(inplace=True)
					])
				else:
					layers.extend([
						nn.Linear(nh, nh),
						activation() if activation != nn.ReLU else activation(inplace=True)
					])
				
		self.model = nn.Sequential(*layers)
	
	def forward(self, x):
		return self.model(x)