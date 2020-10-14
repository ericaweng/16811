import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def main():
    # with open("clear_table.txt") as f:
    with open("cluttered_table.txt") as f:
        Xy = np.array(list(map(lambda x: list(map(float, x.split())), f.read().splitlines())))

    xs = Xy[:, :2]
    zs = Xy[:, 2]
    reg = LinearRegression().fit(xs, zs)
    a, b = reg.coef_
    c = reg.intercept_


    points = np.concatenate((Xy, np.ones((Xy.shape[0], 1))), axis=-1)
    distance = np.abs(points.dot(np.array([a, b, -1, c]))) / np.sqrt(a*a + b*b + 1)
    mean_distance = np.mean(distance)
    print(mean_distance)

    # part 4c
    distance_thresh = .01
    newXy = np.tile((distance < distance_thresh)[...,np.newaxis], (1, Xy.shape[-1])) * Xy
    print(distance < distance_thresh)
    xs = newXy[:, :2]
    zs = newXy[:, 2]
    reg = LinearRegression().fit(xs, zs)
    a, b = reg.coef_
    c = reg.intercept_

    # make data for plotting
    X, Y = np.meshgrid(xs[:, 0], xs[:, 1])
    Z = a * X + b * Y  + c
    print("{:0.3f}x + {:0.3f}y - z = {:0.3f}".format(a, b, -c))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    scatter = ax.scatter(xs[:, 0], xs[:, 1], zs)
    plt.show()

if __name__ == '__main__':
    main()