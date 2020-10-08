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
  P=   np.array([[0.064, 0.265, 0.576],
  [0.692, 0.523, 0.929],
  [0.567, 0.094, 0.319]])
  print(P.mean(axis=1))
  print(P.mean(axis=0))
  # Q=   np.array([[-0.462,  0.992,  0.388],
  # [-1.12,   1.348  ,0.223],
  # [-0.578 , 1.019, -0.19 ]])
  # R =  np.array([[-0.60613632,  0.3061401,  -0.73408243],
  # [-0.76224637 ,-0.48713716,  0.42623686],
  # [-0.22711064 , 0.8179093,   0.5286257 ]])
  # t =   np.array([[-0.09021456],
  # [ 0.63093367],
  # [ 0.01753445]])


  # print(R.dot(P) + t)
  print(Q)


  # P,Q,R,t = (np.array(a) for a in [P,Q,R,t])
  # myR, myt = problem5(P,Q)

  # print(myt.T)
  # print(close_enough?(P.dot(R) + t, Q))
  # print(Q.dot(myR) + myt)

  # main()
