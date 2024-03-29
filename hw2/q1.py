import numpy as np
import matplotlib.pyplot as plt


def table(xs, fxs):
	"""only computes the table, but not the polynomial"""
	assert len(xs) == len(fxs)
	n = len(xs)

	xs = np.array(xs)
	fxs = np.array(fxs)

	def dd(fx0, fx1, x0, x1):
		layer = (fx0 - fx1) / (x0 - x1)
		return layer, x0[:-1], x1[1:]

	table = np.zeros((n, n), dtype=complex)
	table[0] = fxs
	next_x0, next_x1 = xs[:-1], xs[1:]
	for i in range(1, n):
		next_layer, next_x0, next_x1 = dd(table[i-1, i-1:-1], table[i-1, i:], next_x0, next_x1)
		table[i, i:] = next_layer

	return table


def divided_diffs(xs, fxs):
	""" xs are a list of points """
	assert len(xs) == len(fxs)
	n = len(xs)

	def polyx(x):	
		def poly(i, j):
			if not np.isnan(table[i, j]):
				return table[i, j]
			if i == j:
				table[i, j] = fxs[i]
				return fxs[i]
			table[i, j] = ((x - xs[j]) * poly(i, j-1) - (x - xs[i]) * poly(i+1, j)) / (xs[i] - xs[j])
			return table[i, j]

		def poly_vect(i, j):
			if (i, j) in table:
				return table[(i, j)]
			if i == j:
				table[(i, j)] = np.full(x.shape[0], fxs[i])
				return table[(i, j)]
			table[(i, j)] = ( np.multiply((x - xs[j]), poly_vect(i, j-1)) - np.multiply((x - xs[i]), poly_vect(i+1, j))) / (xs[i] - xs[j])
			return table[(i, j)]

		# def poly_vect_d(i, j):
		# 	if (i, j) in table:
		# 		assert (i, j) in dtable
		# 		return table[(i, j)], dtable[(i, j)]
		# 	if i == j:
		# 		table[(i, j)] = np.full(x.shape[0], fxs[i])
		# 		dtable[(i, j)] = np.full(x.shape[0], fxs[i])
		# 		return table[(i, j)], dtable[(i, j)]

		# 	pvd_ij1, dpvd_ij1 = poly_vect_d(i, j-1)
		# 	pvd_i1j, dpvd_i1j = poly_vect_d(i+1, j)
		# 	table[(i, j)] = ((x - xs[j]) * pvd_ij1 - (x - xs[i]) * pvd_i1j) / (xs[i] - xs[j])
		# 	dtable[(i, j)] = ((x - xs[j]) * dpvd_ij1 - (x - xs[i]) * dpvd_i1j) / (xs[i] - xs[j])
		# 	return table[(i, j)], dtable[(i, j)]

		if isinstance(x, float):
			table = np.full((n, n), np.nan)
			ans = poly(0, n-1)
		else:
		# elif False:
			x.shape
			table = {}
			ans = poly_vect(0, n-1)
		# tried to compute derivative fn using divided diffs
		# else:
		# 	x.shape
		# 	table = {}
		# 	dtable = {}
		# 	ans = poly_vect_d(0, n-1)
		return ans

	return polyx

def q1b():
	xs = np.arange(0, .6, 0.125)
	fxs = np.cos(np.pi * xs)
	ans = divided_diffs(xs, fxs)(3/10)
	print("q1b:",  ans)

def indices():
	i = np.arange(1, n+1)
	x_i = i * 2/n - 1
	f_x = f(x_i)

def q1c():
	ns = [2, 4, 40]
	x = 0.07
	f = lambda x: 2 / (1 + 9 * x**2)

	print("q1c:")
	for n in ns:
		i = np.arange(n+1)
		x_i = i * 2/n - 1
		f_x = f(x_i)

		p = divided_diffs(x_i, f_x)
		print("n: %s\t p(%0.2f): %0.9f" % (n, x, p(x)))

	print("Actual f(%0.2f): %0.9f" % (x, f(x)))


def error_est(p_n, f, low=-1, high=1, N=1000):
	"""gets the error estimate for a polynomial estimator p_n 
	and an actual function f over the interval [low, high]"""
	h = 2 / N
	xs = np.arange(low, high + h, h)
	fs = f(xs)
	pns = p_n(xs)
	# plt.plot(xs, fs)
	# plt.plot(xs, pns)
	# plt.show()
	max_En = np.max(np.abs(fs - pns))
	max_En_i = np.argmax(np.abs(fs - pns))
	max_grad_p = np.max(np.abs(np.gradient(pns)))
	return max_En, max_En_i, max_grad_p

def q1d():
	print("q1d:")
	f = lambda x: 2 / (1 + 9 * x**2)
	ns = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
	for n in ns:
		i = np.arange(n+1)
		x_i = i * 2/n - 1
		p = divided_diffs(x_i, f(x_i))
		E_n, En_i, max_grad_p = error_est(p, f)
		# plt.savefig("1d-n-{}.png".format(n))
		# plt.clf() 
		# print("n: {}\tE_n: {:0.2f}".format(n, E_n))
		print("n: {:1.0f}\tE_n: {:0.3f}\tmax grad p: {:0.4f}\tmax grad p * 2 / n: {:0.4f}".format(
			n, E_n, max_grad_p, max_grad_p * 2 / n))

def q1d_resub():
	h = 0.005
	low = -1
	high = 1
	n = 40
	
	xs = np.arange(low, high + h, h)
	f = lambda x: 2 / (1 + 9 * x**2)
	fs = f(xs)
	for i in range(n):
		fs = np.gradient(fs)
		max_grad_f = np.max(np.abs(fs))
		print(max_grad_f)


def main():
	q1d_resub()
	exit()
	divider = "\n"+"#" * 20
	qs = [q1b, q1c, q1d]
	np.set_printoptions(2)
	print(divider)
	for q in qs:
		q()
		print(divider)

	
if __name__ == '__main__':
	main()