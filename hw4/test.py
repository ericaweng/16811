import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

def main2():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	dim = 10

	X, Y = np.meshgrid([-dim, dim], [-dim, dim])
	Z = np.zeros((2, 2))
	angle = .5
	X2, Y2 = np.meshgrid([-dim, dim], [0, dim])
	Z2 = Y2 * angle
	X3, Y3 = np.meshgrid([-dim, dim], [-dim, 0])
	Z3 = Y3 * angle

	r = 7
	M = 1000
	th = np.linspace(0, 2 * np.pi, M)

	x, y, z = r * np.cos(th),  r * np.sin(th), angle * r * np.sin(th)

	ax.plot_surface(X2, Y3, Z3, color='blue', alpha=.5, linewidth=0, zorder=-1)

	# ax.plot(x[y < 0], y[y < 0], z[y < 0], lw=5, linestyle='--', color='green', zorder=0)

	ax.plot_surface(X, Y, Z, color='red', alpha=.5, linewidth=0, zorder=1)

	ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M), lw=5, linestyle='--', color='k', zorder=2)

	ax.plot_surface(X2, Y2, Z2, color='blue', alpha=.5, linewidth=0, zorder=3)

	# ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5, linestyle='--', color='green', zorder=4)
	plt.show()

def main():
	delta = 0.025
	h = np.arange(-2., 2., delta)
	w = np.arange(-2., 2., delta)
	H, W = np.meshgrid(h, w)

	f = H * W
	g = 2 * (H + W)
	df_h = W
	df_w = H
	p = 2
	h = 2-W
	
	fig, ax = plt.subplots()
	# plt.imshow(Z)
	contour_zs = np.arange(-2, 5, .5)
	# CS = ax.contour(H, W, f, contour_zs)
	# CS = ax.contour(H, W, g, [4], colors='red')
	# CS = ax.contour(H, W, df_h, 0, colors='red')
	# CS = ax.contour(H, W, df_w, 0, colors='red')
	# CS = ax.contour(X, Y, Z)

	fig3d = plt.figure()
	ax3d = fig3d.add_subplot(111, projection='3d')
	ax3d.plot_surface(H, W, f, cmap=plt.get_cmap('coolwarm'))
	# print(g[g>p].shape)
	ax3d.plot_surface(H, W, g, cmap=plt.get_cmap('Greens'))

	# ax.plot(h, w, f, lw=5, linestyle='--', color='red', zorder=0)

	# CS = ax3d.plot3D(H, W, color='red')
	# CS = ax3d.contour(H, W, f, [0], colors='red')
	CS = ax3d.contour(H, W, g, [0], colors='red')
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()