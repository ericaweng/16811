import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import scipy    
from scipy.spatial.distance import cdist
from scipy.interpolate import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import prim

# metrics for measurement
## same time

def ema(s, n=1000):
    """
    from: https://cmsdk.com/python/calculate-exponential-moving-average-in-python.html

    returns an n period exponential moving average for
    the time series s
    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer
    returns a numeric array of the exponential
    moving average
    """
    s = np.array(s)
    ema = []
    j = 1
    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)
    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)
    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return ema

def SA(points, init_order, initial_temp, final_temp, alpha):
    # pair_distance = cdist(points, points)
    # assert np.all(np.diag(pair_distance) == 0)
    # k = 2
    # idx = np.argpartition(pair_distance, k+1)[:,1:k+1]
    # print(pair_distance)
    # print(idx)
    # exit()
    # def propose2(order, num_to_switch):
    #     ''' switch a random elem with a random node in the top k adjacent to it'''
    #     new_order = order.copy()
    #     N = new_order.shape[0]
    #     idx_to_switch = np.random.choice(N, size=(num_to_switch,))
    #     for i in idx_to_switch:
    #         # i_switch = idx_to_switch
    #         new_order[i], new_order[i_switch] = new_order[i_switch], new_order[i]
    #     return new_order

    def propose(order, num_to_switch):
        ''' switch a random elem with the node behind-adjacent to it'''
        new_order = order.copy()
        N = new_order.shape[0]
        idx_to_switch = np.random.choice(N, size=(num_to_switch,))
        for i in idx_to_switch:
            new_order[i], new_order[(i+1)%N] = new_order[(i+1)%N], new_order[i]
        return new_order

    current_temp = initial_temp
    old_score = score(points, init_order)
    curr_order = init_order
    N = init_order.shape[0]

    all_orders = [init_order]
    all_scores = []

    best_score = float("inf")
    best_order = None
    best_iter = 0
    i = 0
    while current_temp > final_temp:
        new_order = propose(curr_order, max(N//10, 1))
        new_score = score(points, new_order)
        cost_diff = old_score - new_score
        if cost_diff > 0 or np.random.uniform(0, 1) < np.exp(cost_diff / current_temp):
            all_scores.append(new_score)
            curr_order = new_order
            all_orders.append(new_order)
        if new_score < best_score:
            best_order = curr_order
            best_score = new_score
            best_iter = i
        current_temp *= alpha
        i += 1

    print("SA took:", len(all_scores), "iterations")
    return best_order, best_score, best_iter, all_orders, all_scores

def score(points, order):
    points_in_order = points[order]
    a = np.concatenate([points_in_order[1:], points_in_order[:1]])
    b = points_in_order
    distance = np.linalg.norm(a - b, axis=-1)
    return np.sum(distance, axis=0)

# keep_percent is an integer not a fraction
def cem_opt(f, num_rounds, keep_percent, num_samples):
    keep_count = int(num_samples*keep_percent/100)
    history = []
    for i in range(num_rounds):
        if i == 0:
            order = np.stack([np.random.permutation(N) for _ in range(num_samples)], axis=-1)
        else:
            x_samples = np.random.normal(best_mu, best_stddev, num_samples)
            x_samples = np.clip(x_samples, 0, 1) # Force into a range
        y_samples = f(x_samples)
        best_xs = x_samples[np.argsort(-y_samples)[:keep_count]]  # I want the top scores (hence negative argsort)
        best_mu = np.mean(best_xs)
        best_stddev = np.std(best_xs) + 0.02  # To stop stddev from collapsing too small
        history.append(x_samples)
        
    return best_xs[0], np.array(history)
    

def test_SA(points, initial_temp, final_temp, alpha):
    order = np.random.permutation(N)
    best_order, best_score, best_iter, all_orders, scores = \
        SA(points, order, initial_temp, final_temp, alpha)
    # plot scores of accepted SA proposals
    # ema_scores = ema(scores, 10)
    # plt.plot(np.arange(len(ema_scores)), ema_scores)
    # plt.show()
    print("SA\tbest_order:", best_order, "best_score:", best_score, "best_iter:", best_iter)
    return best_order, best_score

def random_sampling(N, num_samples=1000):
    perms = set()
    while len(perms) < num_samples:
        perms.add(" ".join(map(str, np.random.permutation(N))))
    perms = map(int, perms.split(" "))
    order = np.stack(perms, axis=-1)  # num_samples, N

    sc = score(points, order)
    min_score_i = np.argmin(sc)
    best_order = order[:,min_score_i]
    best_score = sc[min_score_i]
    print("best order:", best_order, "score:", best_score)
    return best_order, best_score
    # return best

def graph_surface(N, num_samples=None):

    # get num_samples permutations, exclude repeats
    # perms = []
    # while len(perms) < num_samples:
    #     perms.append(str(N-1) + " " + " ".join(map(str, np.random.permutation(N-1))))
    # perms = list(map(lambda x: list(map(int, x.split(" "))), list(set(perms))))
    # order = np.array(perms)  # num_samples, N

    def convert_perm_to_edge_set(perm):
        N = len(perm)
        vec = np.zeros((num_samples, N, N))
        edges = set()
        for i in range(N):
            a, b = perm[i], perm[(i+1)%N]
            a, b = max(a, b), min(a, b)
            edges.add("{}{}".format(a, b))
        return " ".join(sorted(edges))

    if num_samples is None:
        num_samples = int(get_total_tours(N))
        # from sympy.utilities.iterables import multiset_permutations
        from itertools import permutations
        # maybe_perms = multiset_permutations(N-1)
        maybe_perms = permutations(np.arange(N-1)) 
        perms_d = {}
        for perm in maybe_perms:
            perm_l = list(perm) + [N-1]
            edge_set = convert_perm_to_edge_set(perm_l)
            perms_d[edge_set] = perm_l
        perms = list(map(lambda x: list(map(int, x.split(" "))), list(perms_d)))
        order = np.array(perms)  # num_samples, N   
        order1 = list(perms_d.values())        
    else:
        assert num_samples > 0
        perms_d = {}
        while len(perms) < num_samples:
            perm = np.random.permutation(N-1).tolist() + [N-1]
            edge_set = convert_perm_to_edge_set(perm)
            perms_d[edge_set] = perm
        perms = list(map(lambda x: list(map(int, x.split(" "))), list(perms_d)))
        order = np.array(perms)  # num_samples, N   
        order1 = list(perms_d.values())

    def convert_order2_to_order1(permss):
        """perms is an iterable where each element looks like: [0 1 5 2 6 4 3]
        returns (number of perms, N) array where 
        each row looks like: [10 30 43 51 52 62 64]"""
        permss = map(convert_perm_to_edge_set, permss)
        permss = list(map(lambda x: list(map(int, x.split(" "))), permss))
        order = np.array(permss) 
        return order

    def convert_order1_to_order2(permss):
        """perms is an iterable where each element looks like: [0 1 5 2 6 4 3]
        returns (number of perms, N) array where 
        each row looks like: [10 30 43 51 52 62 64]"""
        permss = map(convert_perm_to_edge_set, permss)
        permss = list(map(lambda x: list(map(int, x.split(" "))), permss))
        order = np.array(permss) 
        return order

    def convert_order1_to_order2_str(permss):
        """perms is an iterable where each element looks like: [0 1 5 2 6 4 3]
        returns (number of perms, N) array where 
        each row looks like: [10 30 43 51 52 62 64]"""
        permss = map(convert_perm_to_edge_set, permss)
        return list(permss)

    def convert_order_to_one_hot(order):
        num_samples, N = order.shape
        """returns vector of shape (num_samples, N^2) """
        vec = np.zeros((num_samples, (N*(N-1)//2)))
        a, b = order//10, order%10
        edge_i = (a*(a-1))//2 + b 
        indices = np.tile(np.arange(num_samples).reshape(-1,1), (1, N))
        vec[indices, edge_i] = 1
        return vec
    
    def convert_order_to_one_hot2(order):
        num_samples, N = order.shape
        """returns vector of shape (num_samples, N^2) """
        vec = np.zeros((num_samples, N, N))
        for i in range(N):
            a, b = order[:,i], order[:,(i+1)%N]
            print(a, b, num_samples)
            vec[np.arange(num_samples), a, b] = 1
            vec[np.arange(num_samples), b, a] = 1
        i, j = zip(*[(i, j) for i in range(N) for j in range(i)])
        one_hots = vec[:,i,j]
        return one_hots
    
    one_hots = convert_order_to_one_hot(order) 
    # distances between tours
    distances = cdist(one_hots, one_hots, metric="hamming")

    def pca_sorta(distances, alpha=0.01, beta=1, max_epochs=1500):
        """spaces tour points out in 2D euclidean plane"""
        N, _ = distances.shape
        distances = torch.Tensor(distances)  # get the distances squared
        tour_points = nn.Parameter(torch.Tensor(np.random.randn(N, 2)))
        optimizer = optim.Adam([tour_points], lr=alpha)
        for _ in range(max_epochs):
            curr_dist = torch.cdist(tour_points, tour_points)
            error = torch.sum((curr_dist - distances) ** 2) + beta * torch.sum(tour_points**2)  # regularization term
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            if _ % 1000 == 0:
                print("run {}: {}".format(_, error))
        print("finished after {} iters".format(_))
        return tour_points

    tour_points = pca_sorta(distances) 
    tour_points = tour_points.detach().numpy()

    # # plot tour_points in euclidean plane
    # pqs = [(1 + 0.1), (1 - 0.1)]
    # fig, ax = plt.subplots(1, 1)
    # for i, (x, y) in enumerate(tour_points):
    #     ax.plot(x, y, 'bo')
    #     p, q = pqs[i % 2], pqs[(i//2) % 2]
    #     ax.text(x * p, y * q , order[i], fontsize=12)
    # plt.savefig("tour_points.png")
    # plt.clf()

    def score2(points, order):
        i, j = order // 10, order % 10
        a, b = points[i], points[j]
        distance = np.linalg.norm(a - b, axis=-1)
        return np.sum(distance, axis=1)

    # run SA to get the orders
    ##### LALALLALALA
    # _, _, _, all_orders, all_scores = \
    #     SA(points, np.random.permutation(N), initial_temp, final_temp, alpha)

    # graph
    fig = plt.figure(figsize=plt.figaspect(.3))

    # do 2D SA subplot first
    ax = fig.add_subplot(131)
    sa_step = ax.text(low + 1, high - 3, 'SA step: {}'.format(0), fontsize=10)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)   
    ln, = ax.plot(*points.T, 'ro')  # points are unchanging
    # ln, = ax.plot(*points[all_orders[0]].T)  # but the tour changes 

    # now do 3D stuff computations
    # ##### LALALLALALA
    # all_orders2 = convert_order1_to_order2_str(all_orders)
    # dictionary mapping an order to index it is found in order / tour_points
    # e.g.
    d = {" ".join(map(str, point)): i for i, point in enumerate(order)}
    
    ## graph 3D cost surface
    sc = score2(points, order)

    # do 3D TSP tour space cost 
    ax3D = fig.add_subplot(132, projection='3d')
    # ax3D = fig.add_subplot(122)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # low_x, low_y = np.min(tour_points, axis=0)
    # high_x, high_y = np.max(tour_points, axis=0)
    # print("tour_points.shape:", tour_points.shape)

    # Plot the surface.
    surf = ax3D.plot_trisurf(tour_points[:,0], tour_points[:,1], sc, cmap='coolwarm')
    # do contour instead of 3D projection
    # ax3D.contourf(tour_points[:,0], tour_points[:,1], sc, facecolors='coolwarm')

    # initial SA line
    lines = []
    texts = []
    pqs = [(1 + 0.1), (1 - 0.1)] 
    # ind = d[all_orders2[0]]
    # line, = ax3D.plot(*tour_points[ind:ind+1].T, all_scores[0:1], 'ro', zorder=2.5)

    # text = ax.text(tour_points[ind,0] * pqs[i % 2], 
    #                tour_points[ind,0] * pqs[(i//2) % 2], 
    #                all_scores[i],
    #                " ".join(map(str, order[i])), fontsize=12)
    
    # make plot pretty
    ax3D.set_zlabel('tour length')
    ax3D.set_title('simulated annealing')

    # show initial points
    # plt.show()
    # exit()

    def update_lines(num, lines, tour_points, all_orders, all_scores):
        # NOTE: there is no .set_data() for 3 dim data...
        ind = d[all_orders2[num]]
        x, y = tour_points[ind].T
        z = tour_points
        line.set_data(x, y)
        line.set_3d_properties(z)
        # p, q = pqs[i % 2], pqs[(i//2) % 2]
        # text.set_text(x * p, y * q, order[num-k+i])
        # text.set_3d_properties(z)
        sa_step.set_text("SA step: {}".format(num))

        x, y = points[all_orders[num]].T

        # close the tour
        x = x.tolist() + [x[0]]
        y = y.tolist() + [y[0]]
        ln.set_data(x, y)
        return line, ln, sa_step

        return lines, text

    # Creating the Animation object
    # ani = animation.FuncAnimation(fig, update_lines, 25, 
    #         fargs=(lines, tour_points, all_orders, all_scores),
    #         interval=400, blit=False)

    # plt.show()

    # plot clickable thing
    # # plot tour_points in euclidean plane
    pqs = [(1 + 0.1), (1 - 0.1)]
    ax2 = fig.add_subplot(133)
    for i, (x, y) in enumerate(tour_points):
        ax2.plot(x, y, 'bo')
        p, q = pqs[i % 2], pqs[(i//2) % 2]
        # ax2.text(x * p, y * q , order[i], fontsize=12)
    # plt.show()
    # exit()
    prev_plot = None
    while True:
        plt.sca(ax2)
        xc, yc = plt.ginput(1, mouse_stop=2)[0]

        distances = cdist(tour_points, np.array([[xc, yc]]))[:,0]
        closest_point_i = np.argmin(distances)

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2)
        if prev_plot is not None:
            prev_plot.remove()
            prev_plot3D.remove()
            ln.remove()

        tour_x, tour_y = tour_points[closest_point_i]
        prev_plot, = ax2.plot(*tour_points[closest_point_i], 'ro')
        prev_plot3D, = ax3D.plot([tour_x], [tour_y], [sc[closest_point_i]], 'ro', zorder=2.5)
        # print(closest_point_i)
        # print(points)
        # print(order)
        x, y = points[order1[closest_point_i]].T
        # close the tour
        x = x.tolist() + [x[0]]
        y = y.tolist() + [y[0]]
        ln, = ax.plot(x, y, 'b')
        # exit()
        # # draw points
        # pts1.append([x, y])
        # pts2.append([x2, y2])

        plt.draw()

    return


def get_total_tours(N):
    return scipy.special.factorial(N-1)//2

import random

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

N = 6
total_tours = get_total_tours(N)
print("total_tours:", total_tours)
# ## show how number of total tours scales with number of points
# plt.plot(np.arange(N), get_total_tours(np.arange(N)))
# plt.xlabel("number of points")
# plt.ylabel("total number of different tours (log scale)")
# # plt.yscale("log")
# plt.show()
low = -30
high = 30

initial_temp = 90
final_temp = .1
alpha = .999
points = np.random.uniform(low, high, (N, 2))

best_orders = []

fig, ax, surf = graph_surface(N)


exit()




def GD(points, alpha=0.1):
    # max over greedy, starting at each node and going
    N, _ = points.shape
    vertex_weights = nn.Parameter(torch.Tensor([0,1,2,3,4]))
    max_num_steps = 20
    g = prim.Graph(N) 
    # turn points into edges and weights
    edge_weights = cdist(points, points)
    for _ in range(max_num_steps):
        g.graph = torch.Tensor(edge_weights) + vertex_weights
        g.graph[:, np.arange(N)] += torch.Tensor(vertex_weights).reshape(-1, 1)
        g.graph[np.arange(N), np.arange(N)] = 0 
        cost = g.primMST()

# GD(points)


start = time.time()
best_order, best_score = random_sampling(N, 10000)
print("random_sampling took:", time.time() - start)
best_orders.append(best_order)

start = time.time()
best_order, best_score = test_SA(points, initial_temp, final_temp, alpha)
print("SA took:", time.time() - start)
best_orders.append(best_order)

fig, ax = plt.subplots(1, 1)
ax.set_xlim(low, high)
ax.set_ylim(low, high)
for best_order in best_orders:
    ax.plot(*points[best_order].T)
plt.show()

def animate(points, all_orders):
    #### ANIMATION #####
    # set up animation
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)   
    ln, = ax.plot(*points.T, 'ro')
    ln, = ax.plot(*points[all_orders[0]].T)
    text = ax.text(low + 1, high - 3, 'SA step: {}'.format(0), fontsize=10)
    # plot last order of SA
    # ax.plot(*points[all_orders[0]].T)
    # plt.show()

    num_frames_to_show = 10
    interval = len(all_orders) // num_frames_to_show
    some_orders = all_orders[::interval]
    last_one_times = 10
    some_orders = some_orders + [best_order] * last_one_times

    def update(frame):
        ln.set_data(*points[some_orders[frame]].T)
        text.set_text("SA step: {}".format(frame * interval if frame < len(some_orders) - last_one_times else best_iter))
        return ln, text

    anim = animation.FuncAnimation(fig, update, frames=len(some_orders), interval=400)

    # ffmpeg_writer = animation.writers['ffmpeg']
    # writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
    # output_file = "N{}_low{}_high{}_a{}_total{}.mp4".format(N, low, high, alpha, len(scores))
    # anim.save(output_file, writer=writer)
    plt.show()
