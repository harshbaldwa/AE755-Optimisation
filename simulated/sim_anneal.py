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
                dcost = cost_i - cost

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

    from ..common.test_functions import Bohachevsky, himmel
    xmax = 2000
    ymax = 2000
    xbound = [0, xmax]
    ybound = [0, ymax]
    grid = 200
    n = 10
    # sim anneal related parameters
    alpha = 0.98
    pert_max = 200
    Tmax = 1
    Tmin = 0.001
    Markov_no = 10


    layout = random_layout(n, xbound, ybound, grid).layout

#    layout = Layout( np.ones(n)*1000, np.linspace(10, xmax, n), n).layout
    plt.plot(layout[::2], layout[1::2], "b*")
    cost_in = obj(layout, xbound, ybound)
    print("initial_cost = \t"+ str(cost_in))


    xf , cb= annealing(layout, obj, xbound, ybound, Tmax, Tmin,alpha, Markov_no, pert_max )
    print("final cost =   \t"+ str(cb))

    plt.plot(xf[::2], xf[1::2], "ro")
    plt.title(str(cb)+"\t "+str(cost_in))
    plt.savefig("sim_an.png")
    plt.show()
