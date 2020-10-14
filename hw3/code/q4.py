import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def fit(Xy):
    xs = Xy[:, :2]
    zs = Xy[:, 2]
    reg = LinearRegression().fit(xs, zs)
    a, b = reg.coef_
    c = reg.intercept_

    return a, b, c

def get_distance(points, a, b, c):
    return np.abs(points.dot(np.array([a, b, -1, c]))) / np.sqrt(a*a + b*b + 1)

def q4d():
    with open("cluttered_hallway.txt") as f:
        Xy = np.array(list(map(lambda x: list(map(float, x.split())), f.read().splitlines())))
        Xy = np.concatenate((Xy, np.ones((Xy.shape[0], 1))), axis=-1)

    num_points = Xy.shape[0]
    k = 10
    num_planes = 4
    num_iters = 7000
    distance_thresh = .12
    distance_thresh_elim = .05
    inlier_num_thresh = num_points // 6
    print("inlier_num_thresh:", inlier_num_thresh)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    planes = []
    for plane_i in range(num_planes):
        inliers = None
        outliers = None
        best_fit = None
        best_err = np.inf
        distance_save = None
        i = 0
        # for i in range(num_iters):
        while i < num_iters or best_fit is None:
            points_i = np.random.choice(Xy.shape[0], size=k) 
            points = Xy[points_i]
            a, b, c = fit(points)
            distance = get_distance(Xy, a, b, c)
            # mean_distance = np.mean(distance)
            # print("mean_distance:", mean_distance)
            also_inliers = Xy[distance < distance_thresh]
            # print(also_inliers.shape)

            if also_inliers.shape[0] > inlier_num_thresh:
                a, b, c = fit(also_inliers)
                mean_distance = np.mean(get_distance(also_inliers, a, b, c))
                if mean_distance < best_err:
                    # inliers = also_inliers
                    # outliers =  Xy[distance >= distance_thresh]
                    # assert inliers.shape[0] + outliers.shape[0] == Xy.shape[0]
                    distance_save = distance
                    best_fit = a, b, -c
                    best_err = mean_distance
            i += 1
            if i == 10000:
                print("breaking")
                exit()
                break 

        # print plane equation
        plane = "{:0.3f}x + {:0.3f}y - z = {:0.3f}".format(*best_fit)
        print(plane)
        print(best_err)
        planes.append(plane)

        Xy_plane = Xy[distance_save < distance_thresh_elim]
        print("number in this plane:", Xy_plane.shape)

        # make data for plotting
        X, Y = np.meshgrid(Xy_plane[:, 0], Xy_plane[:, 1])
        Z = a * X + b * Y  + c

        Xy = Xy[distance_save > distance_thresh_elim]
        print("number of samples left:", Xy.shape)

        # plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    scatter = ax.scatter(Xy[:, 0], Xy[:, 1], Xy[:, 2])
    plt.savefig("10.png")
    print("\n".join(planes))


def q4c():
    # with open("clear_table.txt") as f:
    with open("cluttered_table.txt") as f:
        Xy = np.array(list(map(lambda x: list(map(float, x.split())), f.read().splitlines())))

    a, b, c = fit(Xy)

    # part 4c
    distance_thresh = .12
    for i in range(50):
        points = np.concatenate((Xy, np.ones((Xy.shape[0], 1))), axis=-1)
        distance = np.abs(points.dot(np.array([a, b, -1, c]))) / np.sqrt(a*a + b*b + 1)
        mean_distance = np.mean(distance)
        print("mean_distance:", mean_distance)
        Xyi = Xy[distance < distance_thresh]
        print("number of samples left:", Xyi.shape)
        a, b, c = fit(Xyi)
        distance_thresh *= .9

    # print plane equation
    print("{:0.3f}x + {:0.3f}y - z = {:0.3f}".format(a, b, -c))

    # make data for plotting
    X, Y = np.meshgrid(Xy[:, 0], Xy[:, 1])
    Z = a * X + b * Y  + c

    # plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    scatter = ax.scatter(Xyi[:, 0], Xyi[:, 1], Xyi[:, 2])
    plt.show()

if __name__ == '__main__':
    q4d()