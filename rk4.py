import numpy as np

def rk4method(func, ic, times):
    N = len(times)
    h = times[1] - times[0]
    w_i = ic
    solution = np.zeros((N, ic.shape[0]))
    solution[0] = ic
    for i in range(1, N):
        t_i = times[0] + h * (i - 1)
        k1 = h * func(t_i, w_i)
        k2 = h * func(t_i + h / 2, w_i + k1 / 2)
        k3 = h * func(t_i + h / 2, w_i + k2 / 2)
        k4 = h * func(t_i + h, w_i + k3)
        w_i = w_i + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        solution[i] = w_i
    return solution