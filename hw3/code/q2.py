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

	num = 12
	ps = list(powerset(bases_fns))[:num]
	ps_names = list(powerset(bases_names))[:num]
	print("powerset size:", len(ps))

	import multiprocessing
	num_cores = multiprocessing.cpu_count() - 10
	print("num_cores:", num_cores)
	pool = multiprocessing.Pool(num_cores)
	args_list = list(zip([[f(x) for f in bases_fns] for bases_fns in ps], ps_names, [points for _ in ps]))
	
	print("done with args_list")
	results = pool.map(opt1, args_list)

	k = 25
	# smallest_i = np.argpartition(results, k, axis=0)[:,0]
	top = sorted(results)[:k]
	# results[smallest_i[:k]]]
	print("".join(["{}\t{}\n".format(loss, equation) for loss, equation in top]))
	import ipdb; ipdb.set_trace()

def opt1(args):
	bases, bases_names, points = args
	basis = torch.FloatTensor(np.stack(bases))
	y = torch.FloatTensor(points)

	coeff = nn.Parameter(torch.FloatTensor(np.zeros(len(bases))))#np.random.randn(len(bases))))

	optimizer = optim.SGD([coeff], lr=.5)

	lamb = 10.
	for _ in range(1000):
		loss = F.mse_loss(y, basis.T.matmul(coeff))# + torch.mean(lamb * (coeff))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_numpy = loss.data.numpy().item()
	equation = " + ".join(["{:0.4f}\\cdot {}".format(c.detach().numpy().item(), bn) for bn, c in zip(bases_names, coeff)])
	# equation = " + ".join(["{:0.4f}\\cdot {}".format(c.detach().numpy().item(), bn) for bn, c in zip(bases_names, coeff)])

	# print("loss:", loss_numpy)
	return loss_numpy, equation


def m():

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

if __name__ == '__main__':
	main()