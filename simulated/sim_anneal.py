import numpy as np
import matplotlib.pyplot as plt


def perturb_state(x, per_max, k):
    per = per_max * (2 * np.random.rand() - 1)
    x[k] = x[k] + per
    return x


def annealing(x0, fn_obj, Tmax, Tmin, alpha, Markov_no, per_max):
    x_best = copy(x0)
    x = x0.copy()
    T = Tmax

    cost = fn_obj(x0)
    cost_best = cost
    xi = np.zeros_like(x0)

    n = len(x0)

    while T > Tmin:
        for i in range(Markov_no):
            for k in range(n):
                xi = perturb_state(x, per_max, k)
                cost_i = fn_obj(xi)
                dcost = cost - cost_i

                if dcost < 0:
                    x = xi.copy()
                    cost = cost_i

                    if cost < cost_best:
                        cost_best = cost
                        x_best = x.copy()

                elif np.exp(-dcost / T) > np.random.rand():
                    x = xi.copy()

        T = alpha * T
    return (x_best, cost_best)
