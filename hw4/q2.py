import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 


def steepest_descent():
	line = lambda xi, grad: lambda x, z: grad * (x - xi[0]) + z # given point and grad, returns you function z
	xi = 1, -1
	# while xi != ():
	for _ in range(3):
		z = fZ(*xi)
		grad = dfZ(*xi)
		l = line(xi, grad)
		point = l(xi, z)
		print (z, grad, l)
	# ax.axis("equal")
	# plt.show()


def main():
	delta = 0.025
	xx = np.arange(-.5, 2.0, delta)
	yy = np.arange(-1.0, 1.0, delta)
	X, Y = np.meshgrid(xx, yy)

	Z = X**3 +  Y ** 3 - 2 * X**2 + 3*Y**2 - 8
	dZx = 3*X**2 + Y**3 - 4*X  + 3*Y **2
	dZy = X**3+ 3*Y**2 - 2* X**2 + 6*Y 

	fZ = lambda x, y: x**3 +  y ** 3 - 2 * x**2 + 3*y**2 - 8
	dfZ = lambda x, y: (3*x**2 + y**3 - 4*x  + 3*y **2, x**3+ 3*y**2 - 2* x**2 + 6*y )

	fig, ax = plt.subplots()
	# plt.imshow(Z)
	# contour_zs = np.arange(-2, 5, .5)
	# CS = ax.contour(X, Y, dZx, contour_zs)
	# CS = ax.contour(X, Y, dZy, contour_zs)
	# CS = ax.contour(X, Y, dZx, 0, colors='red')
	# CS = ax.contour(X, Y, dZy, 0, colors='red')
	# CS = ax.contour(X, Y, Z)
	# fig3d = plt.figure()
	# ax3d = fig3d.add_subplot(111, projection='3d')
	# ax3d.plot_surface(X, Y, Z, cmap=plt.get_cmap('coolwarm'))

	# CS = ax3d.plot3D(X, Y, dZx, color='red')
	# CS = ax3d.contour(X, Y, dZx, [0], colors='red')
	# CS = ax3d.contour(X, Y, dZy, [0], colors='red')
	# plt.tight_layout()
	# plt.show()


def plot2d(path_init, i=None, q_num=None, show=False):
	# plt.plot(path_init[:, 0], path_init[:, 1], 'ro')
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig("{}-{}".format(q_num, i))
	plt.clf()



if __name__ == '__main__':
	main()
