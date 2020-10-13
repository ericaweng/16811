import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

import numpy as np

def main():
	with open("clear_table.txt") as f:
		Xy = np.array(list(map(lambda x: list(map(float, x.split())), f.read().splitlines())))
	xs = Xy[:, :2]
	zs = Xy[:, 2]
	reg = LinearRegression().fit(xs, zs)
	coef = reg.coef_
	print(coef)

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	# Make data.
	X, Y = np.meshgrid(Xy[:, 0], Xy[:, 1])
	Z = coef[0] * X + coef[1] * Y

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	scatter = ax.scatter(*np.split(Xy, Xy.shape[1], axis=1))

	# Customize the z axis.
	# ax.set_zlim(-1.01, 1.01) 
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# A StrMethodFormatter is used automatically
	# ax.zaxis.set_major_formatter('{x:.02f}')

	# Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()

if __name__ == '__main__':
	main()