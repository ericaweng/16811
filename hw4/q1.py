import numpy as np
import matplotlib.pyplot as plt


def main():
    q1d()
    # q1c()

def q1d():
    """ab4 impl"""

    h = 0.05
    x_is = np.arange(0,1+h, h)
    y_i_trues = x_is ** (1/3)

    y_i = 1
    y_is = [y_i]

    f = lambda y: 1 / (3 * y**2)
    y1 = 1.04768955317165
    y2 = 1.03228011545637
    y3 = 1.01639635681485
    y4 = 1.
    for _ in range(len(x_is) - 1):
        y = y4 - h * (55/24*f(y4) - 59/24*f(y3) + 37/24*f(y2) - 9/24*f(y1))
        y_is.append(y)
        y1 = y2
        y2 = y3
        y3 = y4
        y4 = y

    y_is = y_is[::-1]

    print_table(x_is, y_is, y_i_trues)
    plot(x_is, y_i_trues, y_is, q1b(), q1c())


def q1c():
    """runge-kutta 4 impl"""
    # alphas = [0, .5, .5, 1]
    # betas = [.5, .5, 1]

    h = 0.05
    x_is = np.arange(0,1+h, h)
    y_i_trues = x_is ** (1/3)

    y_i = 1
    y_is = [y_i]

    # y = lambda x: x ** (1/3)
    f = lambda y: 1 / (3 * y**2)

    for _ in range(len(x_is)-1):
    #     k_i = y_t + h * np.sum(betas[i] if i * f(k_))
    #     y_t = y_t + h * sum()
        k1 = h * f(y_i)
        k2 = h * f(y_i - 1/2*k1)
        k3 = h * f(y_i - 1/2*k2)
        k4 = h * f(y_i - k3)
        y_i = y_i - (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)
        y_is.append(y_i)

    y_is = y_is[::-1]

    print_table(x_is, y_is, y_i_trues)
    # plot(x_is, y_i_trues, y_is)
    # plot(x_is, y_i_trues, y_is, q1b())
    return y_is

def q1b():
    """euler's method impl"""
    step_sz = 0.05
    x_is = np.arange(0,1+step_sz, step_sz)
    y_i_trues = x_is ** (1/3)

    y_i = 1
    y_is = [y_i]

    y_p = lambda y: 1 / (3 * y**2)

    for _ in range(len(x_is) - 1):
        y_i = y_i - step_sz * y_p(y_i)
        y_is.append(y_i)
    y_is = y_is[::-1]

    # print_table(x_is, y_is, y_i_trues)
    # plot(x_is, y_is, y_i_trues)
    return y_is


def print_table(x_is, y_is, y_i_trues):
    print("""\\begin{center}
\\begin{tabular}{ |c|c|c|c| } 
\\hline""")
    print("x_i & y_i & y(x_i) & y(x_i) - y_i \\\\")
    print("\\hline")
    print("\\\\\n".join(map(lambda tup: "{:0.3f} & {:0.3f} & {:0.3f} & {:0.4f}".format(*tup), zip(x_is, y_is, y_i_trues, y_i_trues - y_is))))
    print("""\\\\\n\\hline
\\end{tabular}
\\end{center}""")


def plot(x_is, y_i_trues, *args):
    plt.plot(x_is, y_i_trues, color='r')
    for arg in args:
        plt.plot(x_is, arg)
    plt.show()


if __name__ == '__main__':
    main()
