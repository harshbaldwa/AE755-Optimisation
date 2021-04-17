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

    k_len = n//5 + 1
    while T > Tmin:
        accepted = 0
        # per_max = per_max * ( 1.0 -0.9*(Tmax-T)/(Tmax-Tmin) )
        for i in range(Markov_no):

            # print("diff" , ( 1-0.9*(Tmax-T)/(Tmax-Tmin) ), per_max)
            for k in range(k_len):
                xi = perturb_state_2(x, per_max , k) # * (0.2 * T / Tmin)
                cost_i = fn_obj(xi, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob)
                dcost = cost_i - cost

                # print("dcost", dcost)
                if dcost < 0:
                    x = xi.copy()
                    cost = cost_i
                    accepted += 1
                    # print("in 1")

                    if cost < cost_best:
                        cost_best = cost
                        x_best = x.copy()
                        print("Current Best", cost_best)

                # elif np.exp(-dcost / T) > np.random.random():
                else:
                    met = np.exp(-dcost / T)
                    rand = np.random.random()
                    # print("dcost", dcost)
                    # print("met {}\t rand {}".format(met, rand))
                    if met > rand:
                        accepted += 1
                        x = xi.copy()
                        # print("in 2")
        accepted_per = accepted*100/(Markov_no*k_len)
        print("Accepted\t{:.3f}%\t pert_max ={:.3f}".format(accepted_per, per_max * ( 1.0 -0.9999*(Tmax-T)/(Tmax-Tmin) )))
        if accepted_per > 55 or accepted_per<45:
            per_max = per_max *( 1 + (accepted_per-50)/100**(-0.5))
            if per_max>xbound[1]:
                per_max = xbound[1]
                print(xbound[1], "xbound")
            elif per_max<100:
                per_max = 100
            print("Current per_max={}".format(per_max))
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

    grid = 100

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
    alpha = 0.82
    pert_max = 150
    Markov_no = 200
    Tmax = 1000000 * np.abs(cost_in)
    Tmin = n*100 #np.abs(cost_in) * 10e-7

    # x2 = perturb_state_2(layout, pert_max, 1)
    # c2 = objective(x2, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob)

    # print(c2)
    # print(x2)

    # sys.stdout = open("Sim_run_Tejas33", "w")
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
    # sys.stdout.close()

    algo_data = [
        "Sim_anneal",
        "Markov: {}\nTmax: {}\nTmin: {}".format(
            Markov_no, Tmax, Tmin
        ),
        "n_turb: {}\ndiameter: {}\nheight: {}\ncost_model: {}\nprofit: ${:.3f}M\ntime: {:.3f}s".format(
            n, diameter, height, 'tejas', -cb / 1e6, b-a
        ),
        "siman{}".format(n),
    ]
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
        algo_data
    )