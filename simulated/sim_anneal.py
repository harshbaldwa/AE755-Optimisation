import numpy as np
import matplotlib.pyplot as plt


def write_run_info(
    x0,
    n,
    bounds,
    Tmax,
    Tmin,
    alpha,
    Markov_no,
    per_max,
    diameter,
    height,
    z_0,
    windspeed_array,
    theta_array,
    wind_prob,
):
    print("#" * 80)
    print("Run parameters")
    print("#" * 80)
    print("N", n)
    print("x\n", x0[::2])
    print("#" * 80)
    print("y\n", x0[1::2])
    print("#" * 80)
    print("bounds\n", bounds)
    print("#" * 80)
    print("diameter", diameter)
    print("height", height)
    print("z0", z_0)
    print("#" * 80)
    print("Tmax", Tmax)
    print("Tmin", Tmin)
    print("alpha", alpha)
    print("Markov no", Markov_no)
    print("per max", per_max)
    print("#" * 80)


def perturb_state(x, per_max, k):
    per = per_max * (2 * np.random.rand() - 1)
    x[k] = x[k] + per
    return x


def perturb_state_2(x, per_max, k):
    per = per_max * (2 * np.random.random(len(x)) - 1)
    x = x + per
    return x


def annealing(
    x0,
    fn_obj,
    bounds,
    Tmax,
    Tmin,
    alpha,
    Markov_no,
    per_max,
    diameter,
    height,
    z_0,
    windspeed_array,
    theta_array,
    wind_prob,
):
    x_best = x0.copy()
    x = x0.copy()
    T = Tmax

    cost = fn_obj(
        x0,
        bounds,
        diameter,
        height,
        z_0,
        windspeed_array,
        theta_array,
        wind_prob,
    )
    cost_best = cost
    xi = np.zeros_like(x0)

    n = len(x0)

    while T > Tmin:
        for i in range(Markov_no):
            for k in range(10):
                xi = perturb_state_2(x, per_max * (0.2 * T / Tmin), k)
                cost_i = fn_obj(
                    xi,
                    bounds,
                    diameter,
                    height,
                    z_0,
                    windspeed_array,
                    theta_array,
                    wind_prob,
                )
                dcost = cost_i - cost

                if dcost < 0:
                    x = xi.copy()
                    cost = cost_i
                    # print("in 1")

                    if cost < cost_best:
                        cost_best = cost
                        x_best = x.copy()
                        print("Current Best", cost_best)

                elif np.exp(-dcost / T) > np.random.random():
                    x = xi.copy()
                    # print("in 2")

        T = alpha * T
    return (x_best, cost_best)


if __name__ == "__main__":

    # from ..common.test_functions import Bohachevsky, himmel
    from ..common.layout import Layout, random_layout
    from ..common.cost import objective

    # from ..common.mosetti_cost import objective
    from ..common.windrose import read_windrose
    from ..common.wake_visualization import get_wake_plots
    import sys
    import time

    plt.style.use("dark_background")

    xmax = 4000
    ymax = 3500
    xbound = [0, xmax]
    ybound = [0, ymax]
    bounds = np.array([xbound, ybound])

    diameter = 82
    height = 80
    z_0 = 0.3
    n = 33

    grid = 50

    # windspeed_array, theta_array, wind_prob = read_windrose()
    windspeed_array = np.array([12])
    theta_array = np.array([0])
    wind_prob = np.array([1])

    # layout = Layout( np.ones(n)*1000, np.linspace(10, xmax, n), n).layout
    layout = random_layout(n, xbound, ybound, grid).layout
    # plt.plot(layout[::2], layout[1::2], "b*")
    cost_in = objective(
        layout,
        bounds,
        diameter,
        height,
        z_0,
        windspeed_array,
        theta_array,
        wind_prob,
    )

    # sim anneal related parameters
    alpha = 0.89
    pert_max = 150
    Markov_no = 200
    Tmax = 10 * np.abs(cost_in)
    Tmin = np.abs(cost_in) * 10e-10

    sys.stdout = open("Sim_run_Tejas33", "w")
    write_run_info(
        layout,
        n,
        bounds,
        Tmax,
        Tmin,
        alpha,
        Markov_no,
        pert_max,
        diameter,
        height,
        z_0,
        windspeed_array,
        theta_array,
        wind_prob,
    )
    print("initial_cost = \t" + str(cost_in))
    a = time.time()
    xf, cb = annealing(
        layout,
        objective,
        bounds,
        Tmax,
        Tmin,
        alpha,
        Markov_no,
        pert_max,
        diameter,
        height,
        z_0,
        windspeed_array,
        theta_array,
        wind_prob,
    )
    b = time.time()
    time_required = b - a
    print("final cost =   \t" + str(cb))
    print("time required", time_required)
    sys.stdout.close()

    get_wake_plots(
        xf[::2],
        xf[1::2],
        bounds,
        diameter,
        height,
        z_0,
        windspeed_array,
        theta_array,
        wind_prob,
        "Sim_anneal " + str(cb),
    )
    # plt.plot(xf[::2], xf[1::2], "ro")
    # plt.title(str(cb)+"\t "+str(cost_in))
    # plt.savefig("sim_an.png")
    # plt.show()
