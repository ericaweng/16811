import numpy as np

import torch
from torch import nn
import torch.optim as optim


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

	f = lambda x: np.sin(x) - 0.5
	p = quad(a, b, c)

	optimizer = optim.SGD([a,b,c], lr=.1)

	x = torch.FloatTensor(np.arange(-np.pi/2, np.pi/2, .000001))

	for _ in range(1000):
		loss = torch.mean((f(x) - p(x))**2)
		# loss = torch.max(torch.abs((f(x) - p(x))))
		print(loss.data.numpy().item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print("p(x) = {0.4f}x^2 + {}x + {}".format(a.data.numpy().item(), b.data.numpy().item(),c.data.numpy().item()))
	print("$L2$-error:",  torch.mean((f(x) - p(x))**2))
	print("$L-\\infty$-error:",  torch.max(torch.abs(f(x) - p(x))))

if __name__ == '__main__':
	main()