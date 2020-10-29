import numpy as np
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def main():
	q = np.array([[1,-3,1,-3,0],
				  [0,1,-3,1, -3],
				  [1,1,-12,0,0],
				  [0,1,1,-12,0],
				  [0,0,1,1,-12]])
	q1 = q[:-1, 1:]
	q2 = q[:-1, np.array([0,2,3,4])]

	

	delta = 0.025
	x = np.arange(-1.0, 4.0, delta)
	y = np.arange(-2.0, 3.0, delta)
	X, Y = np.meshgrid(x, y)
	# Z1 = np.exp(-X**2 - Y**2)
	# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
	# Z = (Z1 - Z2) * 2
	Z = 2 * X**2 + 2 * Y ** 2 - 4 * X - 4 * Y + 3
	Z2 =  X**2 + Y ** 2 + 2 * X * Y - 5*X -3*Y + 4
	fig, ax = plt.subplots()
	CS = ax.contour(X, Y, Z, 0)
	CS2 = ax.contour(X, Y, Z2, 0)
	ax.axis("equal")
	plt.tight_layout()
	x = [1/4 *(4 - np.sqrt(2* (2 + np.sqrt(2* (np.sqrt(5) - 1)) - np.sqrt(10 *(np.sqrt(5) - 1))))), 
	 1/4 *(4 + np.sqrt(2* (2 - np.sqrt(2 *(np.sqrt(5) - 1)) + np.sqrt(10 *(np.sqrt(5) - 1)))))]
	y = [ 1/4* (5 - np.sqrt(5) - np.sqrt(2 *(np.sqrt(5) - 1))),
	 1/4* (5 - np.sqrt(5) + np.sqrt(2 *(np.sqrt(5) - 1)))]
	ax.scatter(x, y, color='r')
	# ax.clabel(CS, inline=1, fontsize=10)
	# ax.set_title('Simplest default with labels')
	plt.show()


def latex_print(arr):
  if len(arr.shape) == 1:
    arr = arr[:,None]
  print('\\begin{pmatrix}')
  for i in arr:
    print(" & ".join(["%2.0f" % j for j in i]), '\\\\') 
  print('\\end{pmatrix}')

if __name__ == '__main__':
	main()