import numpy as np
import matplotlib.pyplot as plt


def perturb_state(x, per_max, k):
    per = per_max * (2 * np.random.rand() - 1)
    x[k] = x[k] + per
    return x


def annealing(x0, fn_obj,xbound, ybound, Tmax, Tmin, alpha, Markov_no, per_max):
    x_best = x0.copy()
    x = x0.copy()
    T = Tmax

    cost = fn_obj(x0, xbound, ybound)
    cost_best = cost
    xi = np.zeros_like(x0)

    n = len(x0)

    while T > Tmin:
        for i in range(Markov_no):
            for k in range(n):
                xi = perturb_state(x, per_max, k)
                cost_i = fn_obj(xi, xbound, ybound)
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

if __name__=="__main__":

    from ..common.layout import Layout, random_layout
    from ..common.cost import obj

    layout = random_layout(10, [0, 2000], [0, 2000], 200).layout

    xf = annealing(layout, obj, [0, 2000], [0, 2000], 1, 0.0001,0.98, 100, 200 )

    plt.plot(xf[::2], xf[1::2], "o")
    plt.show()
