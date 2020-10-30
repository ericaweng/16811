import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 


def steepest_descent():
	delta = 0.025
	xx = np.arange(-1, 2.0, delta)
	yy = np.arange(-3.0, 1.0, delta)
	X, Y = np.meshgrid(xx, yy)
	Z = X**3 + Y ** 3 - 2 * X**2 + 3*Y**2 - 8
	dZx = 3*X**2 - 4*X  
	dZy = 3*Y**2 + 6*Y 

	# fig3d = plt.figure()
	# ax3d = fig3d.add_subplot(111, projection='3d')
	# ax3d.plot_surface(X, Y, Z, cmap=plt.get_cmap('coolwarm'))
	# plt.show()
	# exit()

	fig, ax = plt.subplots()
	# plt.imshow(Z)
	contour_zs = np.arange(-2, 5, .5)
	CSx = ax.contour(X, Y, dZx, 9)#, contour_zs)
	CSy = ax.contour(X, Y, dZy, 9)#, contour_zs)
	plt.clabel(CSx, inline=1, fontsize=10)
	plt.clabel(CSy, inline=1, fontsize=10)
	CS = ax.contour(X, Y, dZx, 0, colors='red')
	CS = ax.contour(X, Y, dZy, 0, colors='red')
	CSf = ax.contour(X, Y, Z, 4, colors='green')
	plt.clabel(CSf, inline=1, fontsize=10)
	x1 = 1/3* (-25 + 4* np.sqrt(34))
	y1 = 3*x1
	ax.scatter([1, x1], [-1, y1], s=10)
	ax.axis('equal')
	plt.tight_layout()
	plt.show()
	exit()

	# line = lambda f, grad: lambda x: f( grad * (x - xi[0]) + z # given point and grad, returns you function z
	def line(f, grad, z, x1, y1):
		def func(x):
			m = grad[1] / grad[0]
			y = m*(x - x1) + y1
			return lambda x: f(x, y), df(x, y)
		return func

	f = lambda x, y: x**3 + y**3 -2*x**2 + 3 * y**2 -8
	df = lambda x, y: (3*x**2 -4*x , 3*y**2 + 6*y)
	# min at (4/3, -2)

	# starting point
	xi = 1, -1
	# while xi != ():
	# while not_close(:
	for _ in range(3):
		# 
		z = f(*xi)
		grad = df(*xi)
		line_f, line_df = line(f, grad, z)
		# xi = (x, y) where line_di = 0
		# repeat until convergence
		
	# ax.axis("equal")
	# plt.show()


def main():
	delta = 0.025
	xx = np.arange(-1, 2.0, delta)
	yy = np.arange(-3.0, 1.0, delta)
	X, Y = np.meshgrid(xx, yy)

	Z = X**3 + Y ** 3 - 2 * X**2 + 3*Y**2 - 8
	dZx = 3*X**2 - 4*X  
	dZy = 3*Y**2 + 6*Y 

	fZ = lambda x, y: x**3 +  y ** 3 - 2 * x**2 + 3*y**2 - 8
	dfZ = lambda x, y: ( 3*X**2 - 4*X, 3*Y**2 + 6*Y )

	fig, ax = plt.subplots()
	# plt.imshow(Z)
	contour_zs = np.arange(-2, 5, .5)
	CS = ax.contour(X, Y, dZx, 9)#, contour_zs)
	CS = ax.contour(X, Y, dZy, 9)#, contour_zs)
	CS = ax.contour(X, Y, dZx, 0, colors='red')
	CS = ax.contour(X, Y, dZy, 0, colors='red')
	# CS = ax.contour(X, Y, Z, 4, colors='green')

	# fig3d = plt.figure()
	# ax3d = fig3d.add_subplot(111, projection='3d')
	# ax3d.plot_surface(X, Y, Z, cmap=plt.get_cmap('coolwarm'))

	# # CS = ax3d.plot3D(X, Y, dZx, color='red')
	# CS = ax3d.contour(X, Y, dZx, [0], colors='red')
	# CS = ax3d.contour(X, Y, dZy, [0], colors='red')

	plt.tight_layout()
	plt.show()


def plot2d(path_init, i=None, q_num=None, show=False):
	# plt.plot(path_init[:, 0], path_init[:, 1], 'ro')
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig("{}-{}".format(q_num, i))
	plt.clf()



if __name__ == '__main__':
	steepest_descent()
	# main()
