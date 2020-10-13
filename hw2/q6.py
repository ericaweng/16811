import numpy as np

def main():
	q = np.array([[1,-3,1,-3,0],
				  [0,1,-3,1, -3],
				  [1,1,-12,0,0],
				  [0,1,1,-12,0],
				  [0,0,1,1,-12]])
	q1 = q[:-1, 1:]
	q2 = q[:-1, np.array([0,2,3,4])]

	print(np.linalg.det(q))
	print(-np.linalg.det(q1) / np.linalg.det(q2))

def latex_print(arr):
  if len(arr.shape) == 1:
    arr = arr[:,None]
  print('\\begin{pmatrix}')
  for i in arr:
    print(" & ".join(["%2.0f" % j for j in i]), '\\\\') 
  print('\\end{pmatrix}')

if __name__ == '__main__':
	main()