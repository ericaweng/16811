import numpy as np
import matplotlib.pyplot as plt


# def def q1b():euler()
def q1b():
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

    print("""\\begin{center}
\\begin{tabular}{ |c|c|c| } 
\\hline""")
    print("x_i & y_i & y(x_i) (truth)")
    print("\\hline")
    print("\\\\\n".join(map(lambda tup: "{} & {} & {} ".format(*tup), zip(x_is, y_is, y_i_trues))))
    print("""\n\\hline
\\end{tabular}
\\end{center}""")

    plt.plot(x_is, y_i_trues, color='r')
    plt.plot(x_is, y_is)
    plt.show()
    return


if __name__ == '__main__':
    q1b()
