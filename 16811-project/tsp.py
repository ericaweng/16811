import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.spatial.distance import cdist

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


def initialize_order():
    order = np.random.permutation(np.arange(N))
    return order 

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
    scores = []

    while current_temp > final_temp:
        new_order = propose(curr_order, max(N//10, 1))
        new_score = score(points, new_order)
        cost_diff = old_score - new_score
        if cost_diff > 0 or np.random.uniform(0, 1) < np.exp(cost_diff / current_temp):
            scores.append(new_score)
            curr_order = new_order
            all_orders.append(new_order)
        current_temp *= alpha

    print("SA took:", len(scores), "iterations")
    return curr_order, all_orders, scores

def score(points, order):
    points_in_order = points[order]
    a = np.concatenate([points_in_order[1:], points_in_order[:1]])
    b = points_in_order
    distance = np.linalg.norm(a - b, axis=1)
    return np.mean(distance)

np.random.seed(0)
N = 10
low = -30
high = 30

initial_temp = 90
final_temp = .1
alpha = .999
points = np.random.uniform(low, high, (N, 2))

order = initialize_order()
final_order, all_orders, scores = SA(points, order, initial_temp, final_temp, alpha)

# plot scores of accepted SA proposals
# ema_scores = ema(scores, 10)
# plt.plot(np.arange(len(ema_scores)), ema_scores)
# plt.show()

# set up animation
fig, ax = plt.subplots(1, 1)
ax.set_xlim(low, high)
ax.set_ylim(low, high)   
ln, = ax.plot(*points[final_order].T)
text = ax.text(low + 1, high - 3, 'SA step: {}'.format(0), fontsize=10)
# plot last order of SA
# ax.plot(*points[all_orders[0]].T)
# plt.show()

num_frames_to_show = 10
interval = len(all_orders) // num_frames_to_show
some_orders = all_orders[::interval]
last_one_times = 10
some_orders = some_orders + some_orders[-1:] * last_one_times

def update(frame):
    ln.set_data(*points[some_orders[frame]].T)
    text.set_text("SA step: {}".format(frame * interval if frame < len(some_orders) - last_one_times else (len(some_orders) - last_one_times - 1) * interval))
    return ln, text

animation = anim.FuncAnimation(fig, update, frames=len(some_orders), interval=400)

# ffmpeg_writer = anim.writers['ffmpeg']
# writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
# output_file = "N{}_low{}_high{}_a{}_total{}.mp4".format(N, low, high, alpha, len(scores))
# animation.save(output_file, writer=writer)
plt.show()
