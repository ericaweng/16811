import numpy as np
import scipy.linalg as sla

from ps1 import print_mats, close_enough
from q2 import svd
from q3 import solve


def problem5(points_in, points_out):
  P = np.array(points_in)
  Q = np.array(points_out)

  p_bar = np.mean(P, axis=1, keepdims=True)
  q_bar = np.mean(Q, axis=1, keepdims=True)

  P_new = P - p_bar
  Q_new = Q - q_bar

  U, _, Vh = np.linalg.svd(P_new.dot(Q_new.T))

  R = Vh.T.dot(U.T)
  t = q_bar - R.dot(p_bar) 

  return R, t

def main(): 
  # Generate random points
  num_test_cases = 10
  num_points = 20
  num_dims = 3

  P = np.random.uniform(0, 2, (num_dims, num_points))
  Ps = np.stack([P for i in range(num_test_cases)])

  # Generate random Rs and ts
  Rs = np.array([sla.orth(mat) for mat in np.random.rand(num_test_cases, num_dims, num_dims)])
  # if round(linalg.det(R)) == -1:
  #     R[3] *= -1
  ts = np.random.rand(num_test_cases, num_dims, 1)

  # Generate Q from R and t
  Qs = np.matmul(Rs, P) + ts

  # test our function
  for i, (p, q) in enumerate(zip(Ps, Qs)):
    print("\n"+"#"*12+"\ntest case %d\n" % i + "#"*12)
    R, t = problem5(p, q)
    assert close_enough(Rs[i], R) and close_enough(ts[i], t), "failed. Rs: {} vs {}\t ts: {} vs. {}".format(Rs[i], R, ts[i], t)
    print("R:{}\nt:{}\npassed".format(R, t))

if __name__ == '__main__':
  main()
