import numpy as np
import scipy.linalg as sla

from ps1 import print_mats, close_enough
from q2 import svd
from q3 import solve


def problem5(points_in, points_out):
  I = np.array(points_in)
  O = np.array(points_out)
  # A = O - I 
  # U, S, Vh = svd(A)
  
  centerI = np.mean(I, axis=0)
  centerO = np.mean(O, axis=0)

  translation = centerO - centerI
  newO = O - centerO
  newI = I - centerI

  # print_mats([newO, centerO, newI, centerI])

  # U_i, S_i, Vh_i = svd(newI)
  # U_o, S_o, Vh_o = svd(newO)
  # print_mats([U_i, S_i, Vh_i])
  # print_mats([U_o, S_o, Vh_o])
  # A = newI.dot()

  # rotation = I^{-1} O = V \frac{1}{\Sigma} U^T O
  # rotation = solve(newI, newO)
  # print(rotation)

  U, s, Vh = np.linalg.svd(newI)
  sigma = s[s > 1e-3]
  k = sigma.shape[0]
  s[:k] = 1/sigma
  D = sla.diagsvd(s, U.shape[0], Vh.shape[0])

  P_1 = Vh.T.dot(D.T).dot(U.T)
  rotation = newO.dot(P_1)
  print("rotation matrix:\n", rotation)
  translation = O - rotation.dot(I) 

  # check
  # print(rotation.dot(I + centerI) + centerO)
  # print(rotation.dot(I) + translation)
  print(rotation.T.dot(rotation)) #np.eye(rotation.shape[0])
  assert close_enough(rotation.dot(I) + translation, O), "test case failed"
  # assert np.all(np.multiply(I, rotation) + translation == O)

def main(): 
  # a = [[1,2,3],[1,2,4],[1,4,2]]
  # b = [[1,-2,1],[4,-3,3],[1,-3,4]]
  PQs = [([[-12, 2, 0], [-8,9,0],[-4,4,0]], [[-4,0,0],[4,2,0],[0,-5,0]]),
  ([[1,5,1],[4,1,9],[2,-1,-1],[6,3,-10]],
    [[-1.633974596,  4.12252235,  2.708308788],[7.294228634, 6.622780076, 0.9659258263],[-0.3660254038 , 0.4482877361 , -2.380139389],[-10.16025404 ,2.544224088, -5.941057286]]),
  ]
  for i, (p, q) in enumerate(PQs):
    print("\n"+"#"*12+"\ntest case %d\n" % i + "#"*12)
    problem5(p, q)


if __name__ == '__main__':
  main()
