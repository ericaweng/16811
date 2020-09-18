import numpy as np

from ps1 import close_enough


def lu(A):
  def swap_rows(A, i, j):
    """swaps row i and j of A"""
    small, large = sorted((i, j))
    swapped_arr = np.concatenate([A[:small], A[large:large+1], A[small+1:large], A[small:small+1], A[large+1:]])
    assert swapped_arr.shape == A.shape
    return swapped_arr

  N, M = A.shape
  P = np.eye(N).astype(np.float32)
  L = np.zeros((N, N)).astype(np.float32)
  U = np.copy(A).astype(np.float32)
  I = np.eye(N).astype(np.float32)

  for ci in range(M):
    # elimination
    pivot = U[ci, ci]
    if pivot == 0: # need to do row changes
      # find max pivot from remaining rows
      arr = [(idx + ci, np.abs(row)) for idx, row in enumerate(U[ci:])] # arr of (row_idx, row_values) tuples
      j = max(arr, key=lambda t: t[1][ci])[0] # get the row with the largest value in the pivot location
      P = swap_rows(P, ci, j)
      L = swap_rows(L, ci, j)
      U = swap_rows(U, ci, j)

    for ri in range(ci+1, N):
      row = U[ri, ci]

      prev_row = U[ci]
      prev_pivot = prev_row[ci]
      factor = row / prev_pivot

      L[ri, ci] = factor
      U[ri] = U[ri] - factor * prev_row

      assert U[ri, ci] == 0

      # print step
      # print("P:\n",P, "\nL:\n",L+I,"\nU:\n",U,"\n")

      assert close_enough((L + I).dot(U), P.dot(A)), \
        "sides !=, difference:\n{}\n{}".format((L + I).dot(U), P.dot(A))

  D = np.diag(U)[:,None]
  U = U / D
  D = np.diag(D[:,0])  
  L = L + I
  # print("P:\n",P, "\nL:\n",L,"\nD:\n", D,"\nU:\n",U)

  return P, L, D, U

def problem1():
  # all test cases are non-singular
  tests = [
            np.array([[10,-10,0],[0,-4,2],[2,0,-5]]),
            np.array([[1,1,0],[1,1,2],[4,2,3]]),
            np.array([[1,2,3],[4,4,5],[-1,0,3]]),
            np.array([[4,-4,9],[9,1,2],[1,0,0]]),
            np.array([[4,4,2,1],[-2,1,-3,3],[1,-4,2,1],[1,1,0,1]]),
            np.array([[-1,3,2,1,1],[0,-2,3,-3,3],[1,0,-4,-2,1],[0,1,1,0,1],[3,1,1,2,-1]])
           ]

  for test_i, arr in enumerate(tests):
    # np.linalg LDU decomposition function
    # P,L,U = sla.lu(arr)
    # pprint(P)
    # pprint(L)
    # pprint(U)
    print('#'*12, "\ntest case %d" % test_i, "\n" + '#'*12)
    P, L, D, U = lu(arr)
    # check that the LDU decomposition == PA
    print("P:\n",P, "\nA:\n", arr, "\nL:\n",L,"\nD:\n", D,"\nU:\n",U,"\n")
    assert close_enough(L.dot(D).dot(U), P.dot(arr)), "test case failed, difference:\n{}\n{}".format(L.dot(D).dot(U), P.dot(arr))

  print("all tests passed")
  

if __name__ == '__main__':
  problem1()