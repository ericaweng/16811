import numpy as np
import scipy
import q1

def close_enough(a, b, eps=1e-7):
  return np.all(np.abs(a - b) < eps)

def any_close_enough(a, b, eps=1e-7):
  return np.any(np.abs(a - b) < eps)

def muller(x0, x1, x2, f, eps=1e-6):
    # x_i+1 = x_i - f(x_i) / f'(x_i)
    i = 0
    while np.abs(f(x2)) > eps:
        table = q1.table([x0, x1, x2], [f(x0), f(x1), f(x2)])
        a = table[2,2]
        b = table[1,2] + table[2,2] * (x2-x1)
        c = table[0,2]

        det = b**2 - 4*a*c
        if det < 0:
            det = det + 0j
        delta1 = 2*c / (b + np.sqrt(det))
        delta2 = 2*c / (b - np.sqrt(det))
        delta = delta2 if np.abs(delta1) > np.abs(delta2) else delta1
        x0 = x1
        x1 = x2
        x2 = x2 - delta
        if i > 10:
            break
        i += 1
    return x2

def main():
    print("{:1.3f}".format(muller(-1, 2, 3, lambda x: x**3 + x + 1)))


if __name__ == '__main__':
  main()