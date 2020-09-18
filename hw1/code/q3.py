import numpy as np
import scipy.linalg as sla

from ps1 import latex_print


def solve(a, b):
  # a = np.array([[1,1,1],[10,2,9],[8,0,7]])
  # b = np.array([3,2,2])
  # b = np.array([1,3,1])
  # a = np.array([[10,-10,0],[0,-4,2],[2,0,-5]])
  # b = np.array([10,2,13]) 

  U, s, Vh = np.linalg.svd(a)
  # x = np.linalg.solve(a, b)
  # print("U:\n", U, "\ns:\n", s, "\nVh:\n", Vh)

  sigma = s[s > 1e-3]
  k = sigma.shape[0]
  s[:k] = 1/sigma
  S = sla.diagsvd(s, U.shape[0], Vh.shape[0])#, k, k)

  # print("U:\n", U[:, :k], "\ns:\n", S, "\nVh:\n", Vh[:k])
  # print("Vh:\n",Vh.T, "\ns:\n", S, "\nVh:\n", U)
  # a = U[:, :k].dot(S).dot(Vh[:k])
  # print(a)

  x = Vh.T.dot(S).dot(U.T).dot(b)

  # print("$$V:")
  # latex_print(Vh.T)
  # print("\\frac{1}{\\Sigma}:")
  # latex_print(S)
  # print("U^T:")
  # latex_print(U.T)
  # print("$$")

  # print("$$ x = V\\frac{1}{\\Sigma} U^Tb =")
  # latex_print(x[:,None])
  # print("$$")
  # latex_print(a.dot(x) - b)

  return x

def check_problem3():
  """check that for the solution x to Ax = b, with b not in A's col space, 
  Ax - b should output a vector perpendicular to A's column space"""
  ax_b = np.array([-2,1,-1])
  v = np.array([1,2,0])
  w = np.array([1,9,7])

  print(v.dot(ax_b))
  latex_print(v)
  latex_print(w)
  print(w.dot(ax_b))


if __name__ == '__main__':
  check_problem3()