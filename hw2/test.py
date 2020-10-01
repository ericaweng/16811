import numpy as np
import sklearn
import scipy

def close_enough(a, b, eps=1e-7):
  return np.all(np.abs(a - b) < eps)
def any_close_enough(a, b, eps=1e-7):
  return np.any(np.abs(a - b) < eps)

def get_paths():
  with open("paths.txt") as f:
    paths = list(map(lambda x: list(map(float, x.split())), f.read().splitlines()))
  return np.stack([paths[::2], paths[1::2]], axis=-1)

def above_fire_center(p):
  """true if trajectory goes above fire center"""
  pass

def below_fire_center(p):
  pass

def in_ring_of_fire(start, end):
  # https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
  x1, y1, _ = start
  x2, y2, _ = end
  x, y = (5, 5)
  r = 1.5

  a = y1 - y2
  b = x2 - x1
  c = -b * y1 - a * x1 
  dist = np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)
  return dist <= 1.5

  # return (x - 5)**2 + (y - 5)**2 <= 1.5**2

def get_interpolated_path(paths, x_0, y_0):
  paths = np.concatenate([paths, np.ones((*paths.shape[0:2], 1))], axis=-1)
  # paths = paths.swapaxes(0, 1)
  # print(paths.shape)
  x0 = np.array([[x_0, y_0, 1]])

  # determine starting point
  distances = scipy.spatial.distance.cdist(paths[:, 0], x0).flatten().argsort()
  # print(distances.shape)

  for i, path_i in enumerate(distances[2:]):
    for j, path_j in enumerate(distances[1:i]):
      for k, path_k in enumerate(distances[:j]):
        # check if first point in triangle 
        indices = np.array([path_i, path_j, path_k])
        # print(paths[indices].shape)
        A = paths[indices].transpose(1,2,0)
        A0 = A[0]
        # determine alpha by solving linear system
        try:
          alpha = np.linalg.solve(A0, x0.T)
        except np.linalg.LinAlgError:
          print("singular... not in triangle")
          continue
        # TODO all points on one side of ring of fire
        # above_fire_center()
        
        # get entire interpolated trajectory, given set alpha
        trajectory = A.dot(alpha).squeeze(-1)
        if close_enough(alpha, 0) or any_close_enough(1 / (alpha + 1e-9), 0) or np.any(alpha < 0):
          print("alpha not good")
          continue

        # confirm interpolation works out for 1st point
        assert close_enough(trajectory[0], x0), "{}\n{}\n{}".format(trajectory[0], x0, alpha)

        def get_interp_path_fn(traj):
          def p(t):
            assert np.all(t <= traj.shape[0]) and np.all(0 <= t)
            int_t = t.astype(np.int)
            # if close_enough(int_t, t):
            #   return traj[int_t, :2]
            # else:
            low_t = np.floor(t).astype(np.int)
            high_t = np.ceil(t).astype(np.int)
            a_low = (t - low_t)[...,np.newaxis].repeat(traj.shape[-1], axis=-1)
            a_high = (high_t - t)[...,np.newaxis].repeat(traj.shape[-1], axis=-1)
            np.set_printoptions(threshold=np.inf)
            assert np.all(np.abs(a_low + a_high - 1 < 1e-6) + np.abs(a_low + a_high < 1e-6)), "\n{}\n{}".format(a_low + a_high)
            interp_path = a_low * traj[low_t] + a_high * traj[high_t]
            return interp_path[:, :2]
          return p

        p = get_interp_path_fn(trajectory) 
        # check all interpolated points not in ring of fire
        # for start, end in zip(trajectory[:-1], trajectory[1:]):

        if np.any([in_ring_of_fire(start, end) for start, end in zip(trajectory[:-1], trajectory[1:])]):
          "goes through ring of fire"
          continue

        return p

  raise RuntimeError("bad starting point, no path available")

def main():
  import matplotlib.pyplot as plt
  from matplotlib import patches

  paths = get_paths()
  p = get_interpolated_path(paths, 2, 3)
  new_path = p(np.arange(0, 49.01, .5))
  print(new_path.shape)
  
  fig, ax = plt.subplots(1)
  ax.plot(*paths.transpose(2,1,0), color='g', linewidth=0.3)
  ax.add_patch(patches.Circle((5, 5), radius=1.5))
  ax.axis("equal")
  ax.plot(*new_path.T, color='r', linewidth=1)
  plt.show()


if __name__ == '__main__':
  main()