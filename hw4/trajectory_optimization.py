import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

waypoints = 300
N = 101
OBST = np.array([[20, 30], [60, 40], [70, 85]])
epsilon = np.array([[25], [20], [30]])

obs_cost = np.zeros((N, N))
for i in range(OBST.shape[0]):
    t = np.ones((N, N))
    t[OBST[i, 0], OBST[i, 1]] = 0
    t_cost = distance_transform_edt(t)
    t_cost[t_cost > epsilon[i]] = epsilon[i]
    t_cost = 1 / (2 * epsilon[i]) * (t_cost - epsilon[i])**2
    obs_cost = obs_cost + t_cost

gx, gy = np.gradient(obs_cost)

SX = 10
SY = 10
GX = 90
GY = 90

traj = np.zeros((2, waypoints))
traj[0, 0] = SX
traj[1, 0] = SY
dist_x = GX-SX
dist_y = GY-SY
for i in range(1, waypoints):
    traj[0, i] = traj[0, i-1] + dist_x/(waypoints-1)
    traj[1, i] = traj[1, i-1] + dist_y/(waypoints-1)

path_init = traj.T
tt = path_init.shape[0]  # number of points

path_init_values = np.zeros((tt, 1))
for i in range(tt):
	# the z-values for our starting path
    path_init_values[i] = obs_cost[int(np.floor(path_init[i, 0])), int(np.floor(path_init[i, 1]))]

# Plot 2D
def plot2d(path_init, q_num=None, i=None, show=False):
	plt.imshow(obs_cost.T)
	plt.plot(path_init[:, 0], path_init[:, 1], 'ro')
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig("{}-{}".format(q_num, i))
	plt.clf()

# Plot 3D
def plot3d(path_init, path_init_values):
	fig3d = plt.figure()
	ax3d = fig3d.add_subplot(111, projection='3d')
	xx, yy = np.meshgrid(range(N), range(N))
	ax3d.plot_surface(xx, yy, obs_cost, cmap=plt.get_cmap('coolwarm'))
	ax3d.scatter(path_init[:, 0], path_init[:, 1], path_init_values, s=20, c='r')
	plt.tight_layout()
	plt.show()

# plot2d(path_init)
# plot3d(path_init, path_init_values)

path = path_init

## 6a gradient descent
def q6a():
	alpha = .1
	for _ in range(4000):
		x, y = np.clip(np.round(path.T[:,1:-1]), 0, N-1).astype(np.int)
		new_path_x = path.T[0,1:-1] - alpha * gx[x, y]
		new_path_y = path.T[1,1:-1] - alpha * gy[x, y]
		path[1:-1] = np.stack([new_path_x, new_path_y]).T
		# print(path)
		path_values = obs_cost[int(np.round(path[i, 0])), int(np.round(path[i, 1]))]
		# plot3d(path, path_values)
	plot2d(path, '6a','4')


def q6b(gx, gy, path):
	alpha = .1
	i = 0
	for _ in range(1000):
		# gx_smooth = gx[1:-1] + np.min(path[1:,0] - path[:-1,0])
		# gy_smooth = gy[1:-1] + np.min(path[1:,1] - path[:-1,1])
		x, y = np.clip(np.round(path.T), 0, N-1).astype(np.int)
		gx_smooth = gx[x, y][1:] + path[1:,0] - path[:-1,0]
		gy_smooth = gy[x, y][1:] + path[1:,1] - path[:-1,1]
		new_path_x = path[1:,0] - alpha * gx_smooth
		new_path_y = path[1:,1] - alpha * gy_smooth
		# don't update start and end point
		path[1:-1] = np.stack([new_path_x, new_path_y]).T[:-1]
		# path_values = obs_cost[np.clip(np.round(path[1:, 0]), 0, N-1).astype(np.int), np.clip(np.round(path[1:, 1]), 0, N-1).astype(np.int)]
		# plot3d(path, path_values)
		if _ == 100 or _ == 200 or _ == 500:
			i += 1
			plot2d(path, i, "6b")
	plot2d(path, i, show=True)

def q6c(gx, gy, path):
	alpha = .1
	i = 0
	for _ in range(5001):
		x, y = np.clip(np.round(path.T), 0, N-1).astype(np.int)
		smoothness = - path[:-2] + 2 * path[1:-1] - path[2:]
		gx_smooth = 0.8 * gx[x, y][1:-1] + 4 * smoothness[:,0]
		gy_smooth = 0.8 * gy[x, y][1:-1] + 4 * smoothness[:,1]
		new_path_x = path[1:-1,0] - alpha * gx_smooth
		new_path_y = path[1:-1,1] - alpha * gy_smooth
		# don't update start and end point
		path[1:-1] = np.stack([new_path_x, new_path_y]).T
		# path_values = obs_cost[np.clip(np.round(path[1:, 0]), 0, N-1).astype(np.int), np.clip(np.round(path[1:, 1]), 0, N-1).astype(np.int)]
		# plot3d(path, path_values)
		if _ == 100 or _ == 5000:
			i += 1
			plot2d(path, i, "6c")
	plot2d(path, i, show=True)


def main():
	# q6c(gx, gy, path)
	q6a()#gx, gy, path)

if __name__ == '__main__':
	main()