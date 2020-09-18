import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

from ps1 import pprint

def svd(a, print_latex=False):
  # a = np.array([[10,-10,0],[0,-4,2],[2,0,-5]])
  # a = np.array([[5,-5,0,0],[5,5,5,0],[0,-1,4,1],[0,4,-1,2],[0,0,2,1]])
  # a = np.array([[1,1,1],[10,2,9],[8,0,7]])
  U, s, Vh = la.svd(a)

  # S = singular values matrix
  S = sla.diagsvd(s, U.shape[0], Vh.shape[0])

  if print_latex:
    print("$$U:")
    latex_print(U)
    print("\\Sigma:")
    latex_print(S)
    print("V^T:")
    latex_print(Vh)
    print("$$")

  # print("Confirm answer\n", np.dot(U, S).dot(Vh))
  return U, S, Vh
  
def lu():
  a = np.array([[10,-10,0],[0,-4,2],[2,0,-5]])
  a = np.array([[5,-5,0,0],[5,5,5,0],[0,-1,4,1],[0,4,-1,2],[0,0,2,1]])
  a = np.array([[1,1,1],[10,2,9],[8,0,7]])
  pl, u = sla.lu(a, True)
  for mat in [pl, u]:
    pprint(mat)
  print("Confirm answer\n", np.dot(pl, u))

def plu():
  a = np.array([[10,-10,0],[0,-4,2],[2,0,-5]])
  a = np.array([[5,-5,0,0],[5,5,5,0],[0,-1,4,1],[0,4,-1,2],[0,0,2,1]])
  a = np.array([[1,1,1],[10,2,9],[8,0,7]])
  P, L, U = sla.lu(a)
  for mat in [P, L, U]:
    pprint(mat)
  print("Confirm answer\n", np.dot(P, L).dot(U))


def main():
  lu()
  plu()
  arrs = [np.array([[10,-10,0],[0,-4,2],[2,0,-5]]),
  np.array([[5,-5,0,0],[5,5,5,0],[0,-1,4,1],[0,4,-1,2],[0,0,2,1]]),
  np.array([[1,1,1],[10,2,9],[8,0,7]])]

  print("\n"+"#"*12,"\n\tsvd:\n"+"#"*12)
  for a in arrs:
    print(svd(a))

if __name__ == "__main__":
  main()
