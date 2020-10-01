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
            assert 0 <= t <= traj.shape[0]
            if close_enough(int(t), t):
              return traj[int(t), :2]
            else:
              low_t = np.floor(t) 
              high_t = np.ceil(t)
              a_low = t - low_t
              a_high = high_t - t
              assert close_enough(a_low + a_high, 1)
              return a_low * p(low_t) + a_high * p(high_t)
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

  point = [(0.8, 1.8), (2.2, 1.0), (2.7, 1.4)]

  paths = get_paths()
  p1 = get_interpolated_path(paths, *point[0])
  p2 = get_interpolated_path(paths, *point[1])
  p3 = get_interpolated_path(paths, *point[2])

  new_path1 = list(zip(*[p1(t) for t in np.arange(0, 49.1, .5)]))
  new_path2 = list(zip(*[p2(t) for t in np.arange(0, 49.1, .5)]))
  new_path3 = list(zip(*[p3(t) for t in np.arange(0, 49.1, .5)]))

  fig, ax = plt.subplots(1)
  ax.plot(*paths.transpose(2,1,0), color='g', linewidth=0.3)
  ax.add_patch(patches.Circle((5, 5), radius=1.5))
  ax.plot(*new_path1, color='m', linewidth=1.3)
  ax.plot(*new_path2, color='b', linewidth=1.3)
  ax.plot(*new_path3, color='r', linewidth=1.3)

  ax.axis("equal")
  plt.show()


if __name__ == '__main__':
  main()