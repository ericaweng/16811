import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import copy


def quad(a, b, c):
	def f(x):
		return a * x ** 2 + b * x + c
	return f

def quad_p(a, b, c):
	def f_p(x):
		return 2*a*x + b
	return f_p

def main():
	a = nn.Parameter(torch.FloatTensor(np.random.randn(1)))
	b = nn.Parameter(torch.FloatTensor(np.random.randn(1)))
	c = nn.Parameter(torch.FloatTensor(np.random.randn(1)))

	a = nn.Parameter(torch.FloatTensor([0.00]))
	# b = nn.Parameter(torch.FloatTensor([.774]))
	# c = nn.Parameter(torch.FloatTensor([-5]))

	f = lambda x: np.sin(x) - 0.5
	p = quad(a, b, c)

	optimizer = optim.SGD([b,c], lr=.001)

	deltas = np.arange(-np.pi/2, np.pi/2, .000001)
	x = torch.FloatTensor(deltas)

	best_loss = np.inf
	for _ in range(12000):
		# loss = torch.mean((f(x) - p(x))**2)
		# loss = torch.max(torch.abs((f(x) - p(x))))
		loss = torch.max(torch.abs((f(x) - p(x)))) #+ a + b + c
		if loss < best_loss:
			best_loss = loss.data.numpy().item()
			best_L2_loss = torch.sqrt(.000001 * torch.sum((f(x) - p(x)) ** 2))
			# best_model = copy.deepcopy(p) # a.data.numpy().item(), b.data.numpy().item(), c.data.numpy().item() 
			print(loss.data.numpy().item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# torch.mean((f(x) - p(x))**2)
	# F.mse_loss
	print("p(x) = {:0.9f}x^2 + {:0.4f}x + {:0.4f}".format(a.data.numpy().item(), b.data.numpy().item(),c.data.numpy().item()))
	print("$L2$-error:", torch.sqrt(.000001 * torch.sum((f(x) - p(x)) ** 2)).data.numpy().item())
	print("$L-\\infty$-error:", torch.max(torch.abs(f(x) - p(x))).data.numpy().item())
	# print("$L2$-error:", torch.sqrt(.000001 * torch.sum((f(x) - best_model(x)) ** 2)).data.numpy().item())
	# print("$L-\\infty$-error:", torch.max(torch.abs(f(x) - best_model(x))).data.numpy().item())
	
	print(best_loss)
	print(best_L2_loss)
	
	import ipdb; ipdb.set_trace()

if __name__ == '__main__':
	main()
