import numpy as np
import matplotlib.pyplot as plt

def perturb_state(x, per_max, n):
    per = per_max*(2*np.random.random(n)-1)
    x = x+per

def annealing(x0, fn_obj, Tmax, Tmin, alpha, Markov_no, per_max):
    x_best = copy(x0)
    x = x0.copy()
    T = Tmax

    cost = fn_obj(x0)
    cost_best = cost
    xi = np.zeros_like(x0)

    n = len(x0)

    while T>Tmin:
        for i in range(Markov_no):
            xi = perturb_state(x, per_max, n)
            cost_i = fn_obj(xi)
            dcost = cost - cost_i

            if dcost<0:
                x = xi.copy()
                if cost_best > cost_i:
                    cost_best = cost_i
                    x_best = x.copy()

            elif np.exp(-dcost/T) > np.random.rand():
                x = xi.copy()

        T = alpha*T
    return(x_best, cost_best)
