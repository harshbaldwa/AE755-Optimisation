# not been tested yet!
# importing is broken, need to fix that
import random
import numpy as np
from ..common import layout
from ..common import cost

class Population:
    # n_pop - number of layouts in a generation (assumed even)
    # x_bound - x bound on the location of turbines
    # y_bound - y bound on the location of turbines
    # N - number of turbines
    # crossover_frac - fraction of top layouts selected for crossover
    # mutation_frac - fraction of total design variables getting mutated
    def __init__(
        self, n_pop, x_bound, y_bound, N, crossover_frac, mutation_frac, grid_size
    ):
        self.n_pop = n_pop
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.pop_matrix = np.zeros((n_pop, 2 * N + 1))
        self.N = N
        self.crossover_pop = np.floor(crossover_frac * n_pop)
        self.mutation_num = int(np.floor(mutation_frac * n_pop))
        self.grid_size = grid_size

    # generate a random population, only for for first generation
    def initialise_random(self):
        for i in range(self.n_pop):
            self.pop_matrix[i, 1:] = layout.random_layout(
                self.N, self.x_bound, self.y_bound, self.grid_size
            ).layout

    # evaluating costs of all layouts and sorting them accordingly
    def fitness_pop(self):
        for i in range(self.n_pop):
            self.pop_matrix[i, 0] = cost.obj(self.pop_matrix[i, 1:], self.x_bound, self.y_bound)

        self.pop_matrix = self.pop_matrix[np.argsort(self.pop_matrix[:, 0])]

    # doing a crossover for generating a new population
    # this is a greedy crossover!!
    def crossover_greedy(self, pop_matrix):
        pop_matrix = np.zeros_like(pop_matrix)
        beta = np.random.random((int(self.n_pop / 2), 2 * self.N))
        parents = np.random.randint(0, self.crossover_pop, self.n_pop)
        for i in range(0, self.n_pop, 2):
            pop_matrix[i, 1:] = (
                beta[int(i / 2)] * (pop_matrix[parents[i], 1:] - pop_matrix[parents[i + 1], 1:])
                - pop_matrix[parents[i], 1:]
            )
            pop_matrix[i + 1, 1:] = (
                beta[int(i / 2)] * (pop_matrix[parents[i + 1], 1:] - pop_matrix[parents[i], 1:])
                - pop_matrix[parents[i + 1], 1:]
            )

        self.pop_matrix = pop_matrix

    # basic mutation (although no idea how effective)
    def mutation(self):
        reshaped_pop = np.reshape(self.pop_matrix[:, 1:], 2 * self.n_pop * self.N)
        mutate_locations = np.random.randint(0, 2 * self.N * self.n_pop, self.mutation_num)
        for location in mutate_locations:
            reshaped_pop[location] = (random.random()) * (
                self.x_bound[1] - self.x_bound[0]
            ) + self.x_bound[0]
        
        self.pop_matrix[:, 1:] = np.reshape(reshaped_pop, (self.n_pop, 2 * self.N))

# just for testing!
pop_size = 30
x_bound = [0, 4000]
y_bound = [0, 3500]
n_turbines = 5
crossover_rate = 0.5
mutation_rate = 0.01
grid_size = 40*5

oldPop = Population(pop_size, x_bound, y_bound, n_turbines, crossover_rate, mutation_rate, grid_size)
oldPop.initialise_random()
oldPop.fitness_pop()

for i in range(20):
    newPop = Population(pop_size, x_bound, y_bound, n_turbines, crossover_rate, mutation_rate, grid_size)
    newPop.crossover_greedy(oldPop.pop_matrix)
    newPop.mutation()
    newPop.fitness_pop()
    print("{}th generation - best cost: {}".format(i+1, newPop.pop_matrix[0, 0]))
    oldPop = newPop
