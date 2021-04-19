import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def annealing(
    x0,
    fn_obj,
    Tmax,
    Tmin,
    alpha,
    Markov_no,
    per_max,
    Ns
):
    x_best = x0.copy()
    x = x0.copy()
    T = Tmax

    cost = fn_obj(x0)
    cost_best = cost
    xi = np.zeros_like(x0)

    n = len(x0)
    pert = per_max*np.ones(n)
    one_hot = np.eye(n)
    accept = np.zeros(n)
    n_iter = int(-np.log(T/Tmin)//np.log(alpha)) +1
    iter_no = 1

    print(n_iter)

    try:
        for _ in tqdm(range(n_iter)):
            for lm in range(Ns):
                for i in range(Markov_no):
                    rand_nos = 2*pert*(2*np.random.random(n) -1 )
                    for k in range(n):
                        xi = x + one_hot[k]* rand_nos
                        cost_i = fn_obj(xi)
                        dcost = cost_i - cost

                        if dcost < 0:
                            x = xi.copy()
                            cost = cost_i
                            accept[k] += 1
                            # print("in 1")

                            if cost < cost_best:
                                cost_best = cost
                                x_best = x.copy()
                                print("Current Best", cost_best, flush = True)
                        elif np.exp(-dcost / T) > np.random.random():
                            accept[k] += 1
                            x = xi.copy()
                            cost = cost_i
                    #end for k
                #end for i (Markov No)

                accepted_per = accept/(Markov_no)
                mh = accepted_per > 0.6
                ml = accepted_per <0.4
                mm = 1 - (mh+ml)
                pert = pert * ((1 + 2*(accepted_per-0.6)/0.4)*mh 
                    + ml/(1 + 2*(0.4-accepted_per)/0.4) + mm)
                # print(accepted_per*100)
                accept.fill(0)
                # print(pert)

            #end for lm 
            iter_no +=1
            T = alpha * T
        #end for _ tqdm()
        print("T = ", T)
        return (x_best, cost_best)
    except KeyboardInterrupt:
        print("\n Oops....\n\tsaving current state")
        restart = np.array([T, x, cost, x_best, cost_best], dtype=object)
        np.save( "restart.npy", restart)
        print("created Restert file")
        return (x_best, cost_best)



if __name__ == "__main__":

    # from ..common.test_functions import  himmel as Test_fn
    from ..common.test_functions import ackely as Test_fn
    # from ..common.test_functions import eggholder as Test_fn
    # from ..common.test_functions import holder_table as Test_fn

    import sys
    import time

    x = (2*np.random.random(6) - 1)*5

    print(x)

    cost_in = Test_fn(x)
    print("initial_cost = \t" + str(cost_in))


    # sim anneal related parameters
    alpha = 0.90
    pert_max = 5
    Markov_no = 100
    Ns = 20
    Tmax = 95000 # np.abs(cost_in)
    Tmin = 10e-10 #np.abs(cost_in) * 10e-15

    #### Start Annealing #####
    a = time.time()
    xf, cb = annealing(
        x,
        Test_fn,
        Tmax,
        Tmin,
        alpha,
        Markov_no,
        pert_max,
        Ns,
    )

    b = time.time()
    time_required = b - a
    print("final cost =   \t" + str(cb))
    print("time required", time_required)
    print("conf:", xf)
    # sys.stdout.close()
