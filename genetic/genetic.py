# cost functions
from numba.np.ufunc import parallel
from ..common import cost as cost
from ..common.windrose import read_windrose

# visualization
from ..common.wake_visualization import get_wake_plots

# other requirements
import numpy as np
from progress.bar import IncrementalBar
from time import time


def init_random(pop, n_pop, bounds, N):
    for i in range(n_pop):
        pop[i, 1::2] = bounds[0, 0] + np.random.random(N) * (
            bounds[0, 1] - bounds[0, 0]
        )
        pop[i, 2::2] = bounds[1, 0] + np.random.random(N) * (
            bounds[1, 1] - bounds[1, 0]
        )


def init_fitness(
    pop, n_pop, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob
):
    for i in range(n_pop):
        pop[i, 0] = cost.objective(
            pop[i, 1:],
            bounds,
            diameter,
            height,
            z_0,
            windspeed_array,
            theta_array,
            wind_prob,
        )

    return pop[np.argsort(pop[:, 0])]

def fitness(
    pop,
    n_pop,
    elit_num,
    bounds,
    diameter,
    height,
    z_0,
    windspeed_array,
    theta_array,
    wind_prob,
):
    for i in range(elit_num, n_pop):
        pop[i, 0] = cost.objective(
            pop[i, 1:],
            bounds,
            diameter,
            height,
            z_0,
            windspeed_array,
            theta_array,
            wind_prob,
        )

    return pop[np.argsort(pop[:, 0])]


def elite(new_pop, old_pop, elit_num):
    new_pop[:elit_num] = old_pop[:elit_num]


def cross(new_pop, old_pop, n_pop, n_var, elit_num, cross_num, tour_size):
    parents_idx = np.random.randint(0, n_pop, (cross_num, tour_size))
    parents_idx.sort(axis=1)
    beta = 0.5 - np.random.random((cross_num, n_var))

    parents1 = old_pop[parents_idx[:, 0], 1:]
    parents2 = old_pop[parents_idx[:, 1], 1:]

    new_pop[elit_num:elit_num+cross_num, 1:] = beta*parents1 + (1-beta)*parents2

    # for i in range(0, cross_num):
    #     parent1 = old_pop[parents_idx[i, 0], 1:]
    #     parent2 = old_pop[parents_idx[i, 1], 1:]
    #     new_pop[elit_num + i, 1:] = beta[i] * parent1 + (1 - beta[i]) * parent2


def mutate(new_pop, old_pop, n_pop, mutat_num, n_var, mutat_gene, b_range, range_par):
    parents_idx = np.random.randint(0, n_pop, mutat_num)
    gene_idx = np.random.randint(1, n_var + 1, (mutat_num, mutat_gene))
    gamma = range_par - 2*range_par * np.random.random((mutat_num, mutat_gene))

    for i in range(0, mutat_num):
        new_pop[-(i + 1), 1:] = old_pop[parents_idx[i], 1:]
        new_pop[-(i + 1), gene_idx[i]] += gamma[i] * b_range[gene_idx[i] % 2]


# turbine data
diameter = 82
height = 80
z_0 = 0.5

# turbines and farm
N = 33
n_var = 2 * N
bounds = np.array([[0, 4000], [0, 3500]])
b_range = np.array([bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]])
# windspeed_array, theta_array, wind_prob = read_windrose()
windspeed_array = np.array([12])
theta_array = np.array([0])
wind_prob = np.array([1])


# optimizer variables
n_pop = 200
elit_num = 24
cross_frac = 0.8
cross_num = int(cross_frac * (n_pop - elit_num))
tour_size = 5
mutat_gene = 2
range_par = 0.5
mutat_num = n_pop - elit_num - cross_num
old_pop = np.zeros((n_pop, n_var + 1))
new_pop = np.zeros((n_pop, n_var + 1))

generations = 100

# start algorithm
init_random(old_pop, n_pop, bounds, N)
old_pop = init_fitness(
    old_pop,
    n_pop,
    bounds,
    diameter,
    height,
    z_0,
    windspeed_array,
    theta_array,
    wind_prob,
)

cross_time = 0
mutate_time = 0
fitness_time = 0

a = time()

try:
    # with IncrementalBar(
    #     "Genetic",
    #     max=generations,
    #     suffix="%(percent).1f%% time_elapsed:[%(elapsed)ds] estimated_time:[%(eta)ds]",
    # ) as bar:
    for gen in range(generations):
        elite(new_pop, old_pop, elit_num)
        d=time()
        cross(new_pop, old_pop, n_pop, n_var, elit_num, cross_num, tour_size)
        e=time()
        mutate(new_pop, old_pop, n_pop, mutat_num, n_var, mutat_gene, b_range, range_par)
        f=time()
        new_pop = fitness(
            new_pop,
            n_pop,
            elit_num,
            bounds,
            diameter,
            height,
            z_0,
            windspeed_array,
            theta_array,
            wind_prob,
        )
        g=time()
        # print("{} - {}, {}, {}".format(gen, new_pop[0, 0], new_pop[1, 0], new_pop[2, 0]))
        # bar.next()
        old_pop = new_pop
        cross_time += e-d
        mutate_time += f-e
        fitness_time += g-f

except KeyboardInterrupt:
    print("Getting the values from last population...\n")

b = time()

print(cross_time, mutate_time, fitness_time)

# algo_data = [
#     "Genetic",
#     "n_pop: {}\nelit_num: {}\ncross: {}\nmutate_gene: {}\nrange_par: {}\ngen: {}".format(
#         n_pop, elit_num, cross_frac, mutat_gene, range_par, generations
#     ),
#     "n_turb: {}\ndiameter: {}\nheight: {}\ncost_model: {}\nprofit: ${:.3f}M\ntime: {:.3f}s".format(
#         N, diameter, height, 'tejas', -new_pop[0, 0] / 1e6, b-a
#     ),
#     "genetic_{}".format(N),
# ]
# print("Profit - ${:.3f}M".format(-new_pop[0, 0] / 1e6))
# get_wake_plots(
#     new_pop[0, 1::2],
#     new_pop[0, 2::2],
#     bounds,
#     diameter,
#     height,
#     z_0,
#     windspeed_array,
#     theta_array,
#     wind_prob,
#     algo_data
# )
