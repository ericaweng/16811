import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def main():
	with open("problem2.txt") as f:
		points = list(map(float, f.read().split()))

	x = np.arange(0, 1.01, .01)
	bases = [1 + 0*x, x, x**2, x**3, x**4,  x**5,  x**6, 
			np.cos(np.pi*x), np.sin(np.pi*x), 
			np.cos(2*np.pi*x), np.sin(2*np.pi*x), 
			np.cos(3*np.pi*x), np.sin(3*np.pi*x),
			np.cos(4*np.pi*x), np.sin(4*np.pi*x),
			np.cos(5*np.pi*x), np.sin(5*np.pi*x)]
	bases_fns = [lambda x:1 + 0*x,lambda x: x,lambda x: x**2,lambda x: x**3,
				lambda x: x**4,lambda x: x**5,lambda x: x**6,
				lambda x: np.cos(np.pi*x), lambda x: np.sin(np.pi*x), 
				lambda x:np.cos(2*np.pi*x), lambda x:np.sin(2*np.pi*x), 
				lambda x:np.cos(3*np.pi*x), lambda x: np.sin(3*np.pi*x),
				lambda x:np.cos(4*np.pi*x), lambda x: np.sin(4*np.pi*x),
				lambda x:np.cos(5*np.pi*x), lambda x: np.sin(5*np.pi*x),
				]
	bases_names = ["1", "x", "x^2", "x^3", "x^4","x^5","x^6", 
					"\\cos(\\pi x)", "\\sin(\\pi x)", "\\cos(2\\pi x)", "\\sin(2\\pi x)", 
					"\\cos(3\\pi x)", "\\sin(3\\pi x)",	"\\cos(4\\pi x)", "\\sin(4\\pi x)",
					"\\cos(5\\pi x)", "\\sin(5\\pi x)"]
	
	def powerset(iterable, start=1):
		"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		import itertools
		s = list(iterable)
		return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1))

	ps = list(powerset(bases_fns))
	ps_names = list(powerset(bases_names))
	# print(len(ps))

	lowest_loss = np.inf
	best_equation = ""

	import multiprocessing
	num_cores = multiprocessing.cpu_count() - 10
	pool = multiprocessing.Pool(num_cores)
	args_list = []
	results = pool.map(f, args_list)

	# def

	for ps_i, (bases_fns, bases_names) in enumerate(zip(ps, ps_names)):
		# if ps_i == 10:
		# 	break

		bases = [f(x) for f in bases_fns]
		basis = torch.FloatTensor(np.stack(bases))
		y = torch.FloatTensor(points)

		coeff = nn.Parameter(torch.FloatTensor(np.zeros(len(bases))))#np.random.randn(len(bases))))

		optimizer = optim.SGD([coeff], lr=0.1)

		lamb = 10.
		for _ in range(1000):
			loss = F.mse_loss(y, basis.T.matmul(coeff))# + torch.mean(lamb * (coeff))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		loss_numpy = loss.data.numpy().item()
		equation = " + ".join(["{:0.4f}\\ {}".format(c.detach().numpy().item(), bn) for bn, c in zip(bases_names, coeff)])
		# equation = " + ".join(["{:0.4f}\\cdot {}".format(c.detach().numpy().item(), bn) for bn, c in zip(bases_names, coeff)])

		print("loss:", loss_numpy)
		if loss_numpy < lowest_loss:
			lowest_loss = loss_numpy 
			best_equation = equation
		print(equation)
		# print("$L2$-error:", torch.mean((f(x) - p(x))**2))
		# print("$L-\\infty$-error:",  torch.max(torch.abs(f(x) - p(x))))

	print()
	print("lowest_loss:", lowest_loss)
	print("best_equation:", best_equation)

if __name__ == '__main__':

	x = np.arange(0, 1.01, .01)
	bases = [1 + 0*x, x, x**2, x**3, x**4,  x**5,  x**6, 
			np.cos(np.pi*x), np.sin(np.pi*x), 
			np.cos(2*np.pi*x), np.sin(2*np.pi*x), 
			np.cos(3*np.pi*x), np.sin(3*np.pi*x),
			np.cos(4*np.pi*x), np.sin(4*np.pi*x),
			np.cos(5*np.pi*x), np.sin(5*np.pi*x)]
	bases_fns = [lambda x:1 + 0*x,lambda x: x,lambda x: x**2,lambda x: x**3,
				lambda x: x**4,lambda x: x**5,lambda x: x**6,
				lambda x: np.cos(np.pi*x), lambda x: np.sin(np.pi*x), 
				lambda x:np.cos(2*np.pi*x), lambda x:np.sin(2*np.pi*x), 
				lambda x:np.cos(3*np.pi*x), lambda x: np.sin(3*np.pi*x),
				lambda x:np.cos(4*np.pi*x), lambda x: np.sin(4*np.pi*x),
				lambda x:np.cos(5*np.pi*x), lambda x: np.sin(5*np.pi*x),
				]
	bases_names = ["1", "x", "x^2", "x^3", "x^4","x^5","x^6", 
	"\\cos(\\pi x)", "\\sin(\\pi x)", "\\cos(2\\pi x)", "\\sin(2\\pi x)", 
	"\\cos(3\\pi x)", "\\sin(3\\pi x)",	"\\cos(4\\pi x)", "\\sin(4\\pi x)",
	"\\cos(5\\pi x)", "\\sin(5\\pi x)"]
	
	bases = np.array(bases)
	print(x.shape)
	print(bases.T.shape)
	bases = bases.T[::10].T
	x = x[::10]
	print(bases.shape, x.shape)
	# exit()

	x, resid, rank, s = np.linalg.lstsq(bases.T, x, rcond=None)
	print(x)
	print(resid)
	print(rank)
	print(s)

	exit(0)
	main()