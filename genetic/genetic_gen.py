# cost functions
from ..common import cost_gen as cost
# from ..common import mosetti_cost_gen as cost
from ..common.windrose import read_windrose

# visualization
from ..common.wake_visualization import get_wake_plots

# other requirements
import numpy as np
from tqdm import tqdm as bar
from time import time

# turbine data
diameter = 82
height = 80
z_0 = 0.3

# turbines and farm
bounds = np.array([[0, 4000], [0, 3500]])

# windspeed_array, theta_array, wind_prob = read_windrose()
windspeed_array = np.array([12])
theta_array = np.array([0])
wind_prob = np.array([1])


N = 26
n_pop = 200
elit_frac = 0.05
cross_frac = 0.8
tour_size = 4
mutat_frac = 0.01


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
    pop[:, 0] = cost.objective(
        pop[:, 1:],
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
    pop[:, 0] = cost.objective(
        pop[:, 1:],
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


def mutate(new_pop, old_pop, n_pop, mutat_num, n_var, mutat_genes, b_range):
    parents_idx = np.random.randint(0, n_pop, mutat_num)
    gene_pool = np.reshape(old_pop[parents_idx, 1:], (mutat_num*n_var))
    gene_idx = np.random.randint(0, mutat_num*n_var, mutat_genes)
    gamma = np.random.random(mutat_genes)

    gene_pool[gene_idx] = gamma*b_range[gene_idx % 2]

    new_pop[n_pop - mutat_num:n_pop, 1:] = np.reshape(gene_pool, (mutat_num, n_var))


b_range = np.array([bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]])
n_var = 2 * N
elit_num = int(elit_frac*n_pop)
cross_num = int(cross_frac * (n_pop - elit_num))
mutat_num = n_pop - elit_num - cross_num
mutat_genes = int(mutat_frac*mutat_num*n_var)
old_pop = np.zeros((n_pop, n_var + 1))
new_pop = np.zeros((n_pop, n_var + 1))

generations = 100*N

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

a = time()

try:
    for gen in bar(range(generations)):
        elite(new_pop, old_pop, elit_num)
        cross(new_pop, old_pop, n_pop, n_var, elit_num, cross_num, tour_size)
        mutate(new_pop, old_pop, n_pop, mutat_num, n_var, mutat_genes, b_range)
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
        # print("{} - {}, {}, {}".format(gen, new_pop[0, 0], new_pop[1, 0], new_pop[2, 0]))
        old_pop = new_pop

except KeyboardInterrupt:
    print("Getting the values from last population...\n")

b = time()

algo_data = [
        "Genetic",
        "n_pop: {}\nelit_frac {}\ncross: {}\ntour_size: {}\nmutat_frac: {}\ngen: {}".format(
        n_pop, elit_frac, cross_frac, tour_size, mutat_frac, generations
        ),
        "n_turb: {}\ndiameter: {}\nheight: {}\ncost_model: {}\nprofit: ${:.3f}M\ntime: {:.3f}s".format(
        N, diameter, height, 'tejas', -new_pop[0, 0]/1e6, b-a
        ),
        "genetic/N_{}".format(N),
]
# print("Profit - {}".format(new_pop[0, 0]))
print("Profit - ${:.3f}M".format(-new_pop[0, 0]/1e6))
get_wake_plots(
        new_pop[0, 1::2],
        new_pop[0, 2::2],
        bounds,
        diameter,
        height,
        z_0,
        windspeed_array,
        theta_array,
        wind_prob,
        algo_data
)
