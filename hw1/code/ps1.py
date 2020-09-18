import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)  


def close_enough(A, B, eps=1e-5):
  """returns true if two matrices A, B are close enough else false"""
  return np.all(np.abs(A - B) < eps)

def pprint(arr):
  for i in arr:
    for j in i:
      print("%2.1f" % j, end=" ")
    print()
  print()

def latex_print(arr):
  if len(arr.shape) == 1:
    arr = arr[:,None]
  print('\\begin{pmatrix}')
  for i in arr:
    print(" & ".join(["%2.2f" % j for j in i]), '\\\\') 
  print('\\end{pmatrix}')


def print_mats(vars, names=None):
  if names is not None:
    print("".join(["\n{}:\n{}".format(name, var) for name, var in zip(names, vars)]))
  else:
    print("".join(["\n{}".format(var) for var in vars]))


