import numpy as np


def newton(x_0, f, f_p, eps=1e-3):
    # x_i+1 = x_i - f(x_i) / f'(x_i)
    x_i = x_0
    while np.abs(f(x_i)) > eps:
        x_i = x_i - f(x_i) / f_p(x_i)
    return x_i


def psyched_up_newton(x_0, f, f_p, eps=1e-3):
    # x_i+1 = x_i - f(x_i) / f'(x_i)
    while np.isnan(f_p(x_0)) or f(x_0) / f_p(x_0) > 1 / eps:
        print("x_0:", x_0)
        x_0 += eps
    x_i = x_0
    last_fx_i = f(x_0)
    alpha = 1
    i = 0
    while np.abs(f(x_i)) > eps:
        x_i = x_i - alpha * f(x_i) / f_p(x_i)
        print("x_i:", x_i, "f(x_i):", f(x_i), "alpha:", alpha)
        if np.abs(f(x_i)) > np.abs(last_fx_i): # lr adjust
            alpha = max(eps, alpha / 20)
            print(alpha)
        else: 
            alpha = min(1, alpha * 2)
        last_fx_i = f(x_i)
        i += 1
        if i > 10000:
            print("exiting")
            break
    return x_i


def main():
    qs = [14, 17.2]
    f = lambda x: np.tan(x) - x
    f_p = lambda x: 1 / np.cos(x) ** 2 - 1
    for q in qs:
        root = newton(q, f, f_p)
        print(root)

    
if __name__ == '__main__':
    main()