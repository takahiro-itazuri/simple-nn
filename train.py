import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, required=True, help='and | or | xor')
parser.add_argument('--act_type', type=str, required=True, help='relu | sigmoid | tanh')
parser.add_argument('--nh', type=int, nargs='*', default=[2,3,10,25], help='number of neurons in hidden layer')
parser.add_argument('--nl', type=int, nargs='*', default=[1,3,10], help='number of hidden layers')
parser.add_argument('--use_bn', action='store_true', default=False, help='use batch normalization')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
opt = parser.parse_args()
num_epochs = 100
checkpoint = 10
batch_size = 4000
os.makedirs(opt.log_dir, exist_ok=True)
filename = os.path.join(opt.log_dir, '{}_{}'.format(opt.data_type, opt.act_type))
if opt.use_bn:
	filename += '_bn'

# data
npoints = 1000
data = torch.FloatTensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]]).repeat(npoints, 1)
dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
if opt.data_type == 'and':
	target = torch.LongTensor([1, 0, 0, 0]).repeat(npoints)
elif opt.data_type == 'or':
	target = torch.LongTensor([1, 1, 1, 0]).repeat(npoints)
elif opt.data_type == 'xor':
	target = torch.LongTensor([0, 1, 1, 0]).repeat(npoints)

# activation
if opt.act_type == 'relu':
	activation_fn = nn.ReLU
elif opt.act_type == 'sigmoid':
	activation_fn = nn.Sigmoid
elif opt.act_type == 'tanh':
	activation_fn = nn.Tanh

# figure
fig = plt.figure()

for i, nl in enumerate(opt.nl):
	for j, nh in enumerate(opt.nh):
		print('========================================')
		print('nl: {:d}, nh: {:d}'.format(nl, nh))
		print('----------------------------------------')

		# model
		model = Model(nh, nl, activation_fn, opt.use_bn)

		# optimizer
		optimizer = optim.Adam(model.parameters(), lr=opt.lr)

		# loss function
		ce_loss = nn.CrossEntropyLoss()

		# train
		model.train()
		for epoch in range(1, num_epochs+1):
			total = 0
			correct = 0
			for itr, (raw) in enumerate(loader):
				optimizer.zero_grad()
				x = raw[0]
				t = target[itr*batch_size:(itr+1)*batch_size]
				y = model(x + 0.25 * torch.randn_like(x))
				loss = ce_loss(y, t)
				loss.backward()
				optimizer.step()

				_, pred = y.max(1)
				total += x.size(0)
				correct += pred.eq(t).sum().item()

			if epoch % checkpoint == 0:
				print('{:d}: acc {:.4f}, loss: {:.2e}'.format(epoch, float(correct)/float(total), loss.item()/float(total)))

		# test
		model.eval()
		grid_size = 1000
		x = np.linspace(-2.0, 2.0, grid_size)
		y = np.linspace(-2.0, 2.0, grid_size)
		xx, yy = np.meshgrid(x, y)
		np_grid = np.c_[xx.ravel(), yy.ravel()]
		torch_grid = torch.from_numpy(np_grid.astype(np.float32))
		# output = model(torch_grid).cpu().detach().numpy()
		# output = np.reshape(output, (grid_size, grid_size, 2))
		# result = output[:, :, 1] - output[:, :, 0]

		softmax_output = F.softmax(model(torch_grid)).cpu().detach().numpy()
		softmax_output = np.reshape(softmax_output, (grid_size, grid_size, 2))
		softmax_result = softmax_output[:, :, 1] - softmax_output[:, :, 0]

		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# surf = ax.plot_surface(xx, yy, result, cmap='seismic', linewidth=0)
		# maxval = np.max(np.abs(result)).item()
		# surf.set_clim(-maxval, maxval)
		# fig.colorbar(surf)
		# plt.savefig(filename + '.png')

		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# surf = ax.plot_surface(xx, yy, softmax_result, cmap='seismic', linewidth=0)
		# surf.set_clim(-1.0, 1.0)
		# fig.colorbar(surf)
		# plt.savefig(filename + '_softmax.png')

		ax = fig.add_subplot(len(opt.nl), len(opt.nh), i * len(opt.nh) + j + 1)
		region = plt.imshow(softmax_result, cmap='seismic', origin='lower')
		region.set_clim(-1.0, 1.0)
		region.axes.get_xaxis().set_visible(False)
		region.axes.get_yaxis().set_visible(False)
		ax.set_title('nl {:d}, nh {:d}'.format(nl, nh), fontdict={'fontsize': 10})

plt.tight_layout()
plt.savefig(filename + '.png')