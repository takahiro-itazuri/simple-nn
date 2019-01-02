import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn, optim
from torch.nn import functional as F

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, required=True, help='AND | OR | XOR')
parser.add_argument('--act_type', type=str, required=True, help='relu | sigmoid')
parser.add_argument('--nhidden', type=int, default=2, help='number of neurons in hidden layer')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
opt = parser.parse_args()
num_epochs = 10000
checkpoint = 1000
os.makedirs(opt.log_dir, exist_ok=True)
filename = os.path.join(opt.log_dir, '{}_{}_nhidden{:03d}'.format(opt.data_type, opt.act_type, opt.nhidden))

# data
data = torch.FloatTensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
if opt.data_type == 'AND':
	target = torch.LongTensor([1, 0, 0, 0])
elif opt.data_type == 'OR':
	target = torch.LongTensor([1, 1, 1, 0])
elif opt.data_type == 'XOR':
	target = torch.LongTensor([0, 1, 1, 0])

# model
if opt.act_type == 'relu':
	activation_fn = nn.ReLU(inplace=True)
elif opt.act_type == 'sigmoid':
	activation_fn = nn.Sigmoid()
model = Model(opt.nhidden, activation_fn)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# loss function
ce_loss = nn.CrossEntropyLoss()

# train
model.train()
for epoch in range(1, num_epochs+1):
	optimizer.zero_grad()
	output = model(data)
	loss = ce_loss(output, target)
	loss.backward()
	optimizer.step()

	_, prediction = output.max(1)
	total = data.size(0)
	correct = prediction.eq(target).sum().item()

	if epoch % checkpoint == 0:
		print('{:d}: acc {:.4f}, loss: {:.4f}'.format(epoch, float(correct)/float(total), loss.item()/float(total)))

# test
model.eval()
grid_size = 1000
x = np.linspace(-2.0, 2.0, grid_size)
y = np.linspace(-2.0, 2.0, grid_size)
xx, yy = np.meshgrid(x, y)
np_grid = np.c_[xx.ravel(), yy.ravel()]
torch_grid = torch.from_numpy(np_grid.astype(np.float32))

output = model(torch_grid).detach().numpy()
output = np.reshape(output, (grid_size, grid_size, 2))
result = output[:, :, 1] - output[:, :, 0]

softmax_output = F.softmax(model(torch_grid)).detach().numpy()
softmax_output = np.reshape(softmax_output, (grid_size, grid_size, 2))
softmax_result = softmax_output[:, :, 1] - softmax_output[:, :, 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, result, cmap='seismic', linewidth=0)
maxval = np.max(np.abs(result)).item()
surf.set_clim(-maxval, maxval)
fig.colorbar(surf)
plt.savefig(filename + '.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, softmax_result, cmap='seismic', linewidth=0)
surf.set_clim(-1.0, 1.0)
fig.colorbar(surf)
plt.savefig(filename + '_softmax.png')

fig = plt.figure()
region = plt.imshow(softmax_result, cmap='seismic')
region.set_clim(-1.0, 1.0)
fig.colorbar(region)
region.axes.get_xaxis().set_visible(False)
region.axes.get_yaxis().set_visible(False)
plt.savefig(filename + '_softmax_region.png')