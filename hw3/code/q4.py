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

def main():
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

    # make data for plotting
    X, Y = np.meshgrid(Xy[:, 0], Xy[:, 1])
    Z = a * X + b * Y  + c
    print("{:0.3f}x + {:0.3f}y - z = {:0.3f}".format(a, b, -c))
    # exit()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    scatter = ax.scatter(Xyi[:, 0], Xyi[:, 1], Xyi[:, 2])
    plt.show()

if __name__ == '__main__':
    main()