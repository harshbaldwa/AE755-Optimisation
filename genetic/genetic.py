# cost functions
from ..common import cost as cost
from ..common.windrose import read_windrose
# visualization
from ..common.wake_visualization import get_wake_plots
#other requirements
import numpy as np
from progress.bar import IncrementalBar



def init_random(pop, n_pop, bounds, N):
    for i in range(n_pop):
        pop[i, 1::2] = bounds[0, 0] + np.random.random(N)*(bounds[0, 1] - bounds[0, 0])
        pop[i, 2::2] = bounds[1, 0] + np.random.random(N)*(bounds[1, 1] - bounds[1, 0])

def init_fitness(pop, n_pop, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob):
    for i in range(n_pop):
        pop[i, 0] = cost.objective(pop[i, 1:], bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob)
    
    return pop[np.argsort(pop[:, 0])]

def fitness(pop, n_pop, elit_num, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob):
    for i in range(elit_num, n_pop):
        pop[i, 0] = cost.objective(pop[i, 1:], bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob)

    return pop[np.argsort(pop[:, 0])]

def elite(new_pop, old_pop, elit_num):
    new_pop[:elit_num] = old_pop[:elit_num]

def cross(new_pop, old_pop, n_pop, n_var, elit_num, cross_num, tour_size):
    parents_idx = np.random.randint(0, n_pop, (cross_num, tour_size))
    parents_idx.sort(axis=1)
    beta = 0.5 - np.random.random((cross_num, n_var))

    for i in range(0, cross_num):
        parent1 = old_pop[parents_idx[i, 0], 1:]
        parent2 = old_pop[parents_idx[i, 1], 1:]
        new_pop[elit_num+i, 1:] = beta[i]*parent1 + (1-beta[i])*parent2

def mutate(new_pop, old_pop, n_pop, mutat_num, n_var, mutat_gene, b_range):
    parents_idx = np.random.randint(0, n_pop, mutat_num)
    gene_idx = np.random.randint(1, n_var+1, (mutat_num, mutat_gene))
    gamma = 0.3 - 0.6*np.random.random((mutat_num, mutat_gene))

    for i in range(0, mutat_num):
        new_pop[-(i+1), 1:] = old_pop[parents_idx[i], 1:]
        new_pop[-(i+1), gene_idx[i]] += gamma[i]*b_range[gene_idx[i]%2]


# turbine data
diameter = 82
height = 80
z_0 = 0.3

# turbines and farm
N = 33
n_var = 2*N
bounds = np.array([[0, 4000], [0, 3500]])
b_range = np.array([bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]])
# windspeed_array, theta_array, wind_prob = read_windrose()
windspeed_array = np.array([12])
theta_array = np.array([0])
wind_prob = np.array([1])


# optimizer variables
n_pop = 150
elit_num = 12
cross_frac = 0.8
cross_num = int(cross_frac*(n_pop-elit_num))
tour_size = 5
mutat_gene = 2
mutat_num = n_pop - elit_num - cross_num
old_pop = np.zeros((n_pop, n_var + 1))
new_pop = np.zeros((n_pop, n_var + 1))

generations = 600


# start algorithm
init_random(old_pop, n_pop, bounds, N)
old_pop = init_fitness(old_pop, n_pop, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob)

try :
    with IncrementalBar('Genetic', max=generations, suffix='%(percent).1f%% time_elapsed:[%(elapsed)ds] estimated_time:[%(eta)ds]') as bar:
        for gen in range(generations):
            elite(new_pop, old_pop, elit_num)
            cross(new_pop, old_pop, n_pop, n_var, elit_num, cross_num, tour_size)
            mutate(new_pop, old_pop, n_pop, mutat_num, n_var, mutat_gene, b_range)
            new_pop = fitness(new_pop, n_pop, elit_num, bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob)
            # print(gen, " - ", new_pop[0:3, 0])
            bar.next()
            old_pop = new_pop

except KeyboardInterrupt:
    print("Getting the values from last population...\n")


print('Profit - ${:.2f}M'.format(-new_pop[0, 0]/1e6))
get_wake_plots(new_pop[0, 1::2], new_pop[0, 2::2], bounds, diameter, height, z_0, windspeed_array, theta_array, wind_prob, "Genetic, Profit: ${:.2f}M".format(-new_pop[0, 0]/1e6))


