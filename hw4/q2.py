import numpy as np
import matplotlib.pyplot as plt


def main():
	delta = 0.025
	x = np.arange(-.5, 2.0, delta)
	y = np.arange(-1.0, 1.0, delta)
	X, Y = np.meshgrid(x, y)

	Z = X**3 +  Y ** 3 - 2 * X**2 + 3*Y**2 - 8
	dZx = 3*X**2 + Y**3 - 4*X  + 3*Y **2
	dZy = X**3+ 3*Y**2 - 2* X**2 + 6*Y 
	fig, ax = plt.subplots()
	CS = ax.contour(X, Y, dZx, 0)
	CS = ax.contour(X, Y, dZy, 0)
	ax.axis("equal")
	# ax.clabel(CS, inline=1, fontsize=10)
	# ax.set_title('Simplest default with labels')
	plt.show()

if __name__ == '__main__':
	main()
