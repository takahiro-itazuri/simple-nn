import torch
from torch import nn

class Model(nn.Module):
	def __init__(self, nhidden, activation='relu'):
		super(Model, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(2, nhidden),
			nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid(),
			nn.Linear(nhidden, nhidden),
			nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid(),
			nn.Linear(nhidden, 2)
		)
	
	def forward(self, x):
		return self.model(x)