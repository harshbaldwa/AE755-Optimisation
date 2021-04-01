# not been tested yet!
# importing is broken, need to fix that
from ..common import mosetti_cost
import random
import numpy as np
from ..common import layout
from ..common.wake_visualization import get_wake_plots
import time
import matplotlib.pyplot as plt
import sys


class Population:
    # n_pop - number of layouts in a generation (assumed even)
    # x_bound - x bound on the location of turbines
    # y_bound - y bound on the location of turbines
    # N - number of turbines
    # crossover_frac - fraction of top layouts selected for crossover
    # mutation_frac - fraction of total design variables getting mutated
    def __init__(
        self,
        n_pop,
        x_bound,
        y_bound,
        N,
        elitism_num,
        crossover_frac,
        mutation_frac,
        grid_size,
    ):
        self.n_pop = n_pop
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.pop_matrix = np.zeros((n_pop, 2 * N + 1))
        self.N = N
        # self.elitism_pop = int(np.floor(elitism_frac * n_pop))
        self.elitism_pop = int(elitism_num)
        self.crossover_pop = int(np.floor(crossover_frac * n_pop))
        self.mutation_num = int(np.floor(mutation_frac * n_pop * 2 * N))
        self.grid_size = grid_size

    # generate a random population, only for for first generation
    def initialise_random(self):
        for i in range(self.n_pop):
            # Y = np.linspace(y_bound[0], y_bound[1], self.N)
            # Y = np.random.random(self.N)
            # X = np.ones(self.N)*1000
            # X = np.linspace(x_bound[0], x_bound[1], self.N)
            # X = 2000*np.random.random(self.N)
            # Y = np.ones(self.N)*1000
            # self.pop_matrix[i, 1:] = layout.Layout(X, Y, self.N).layout
            # self.pop_matrix[i, 1:] = layout.random_layout(
            #     self.N, self.x_bound, self.y_bound, self.grid_size
            # ).layout

            self.pop_matrix[i, 1:] = 2000 * np.random.random(2 * self.N)
        # self.pop_matrix[:, 1:] = np.load("random_pop.npy")

    def purge(self, purge_pop_ratio):
        for i in range(
            int(self.n_pop * purge_pop_ratio), self.n_pop - self.crossover_pop
        ):
            self.pop_matrix[i, 1:] = 2000 * np.random.random(2 * self.N)

    # evaluating costs of all layouts and sorting them accordingly
    def fitness_pop(self, cost_value, power_value):
        for i in range(self.n_pop):
            # self.pop_matrix[i, 0] = cost.obj(self.pop_matrix[i, 1:], self.x_bound, self.y_bound)
            (
                self.pop_matrix[i, 0],
                cost_value[i],
                power_value[i],
                penalty_value[i],
            ) = mosetti_cost.mosetti(self.pop_matrix[i, 1:], self.x_bound, self.y_bound)
            # self.pop_matrix[i, 0] = test_functions.mosetti(self.pop_matrix[i, 1:], self.x_bound, self.y_bound)

        self.pop_matrix = self.pop_matrix[np.argsort(self.pop_matrix[:, 0])]

    def elitism(self, pop_matrix):
        self.pop_matrix[: self.elitism_pop] = pop_matrix[: self.elitism_pop]

    # doing a crossover for generating a new population
    # this is a greedy crossover!!
    def crossover_greedy(self, pop_matrix):
        new_pop_matrix = np.zeros((self.n_pop - self.elitism_pop, 2 * self.N + 1))
        beta = (
            np.random.random((int(self.n_pop - self.elitism_pop / 2), 2 * self.N)) / 5
        )
        parents = np.random.randint(
            0, self.crossover_pop, self.n_pop - self.elitism_pop
        )
        for i in range(0, self.n_pop - self.elitism_pop, 2):
            new_pop_matrix[i, 1:] = (
                beta[int(i / 2)]
                * (pop_matrix[parents[i + 1], 1:] - pop_matrix[parents[i], 1:])
                + pop_matrix[parents[i], 1:]
            )
            new_pop_matrix[i + 1, 1:] = (
                beta[int(i / 2)]
                * (pop_matrix[parents[i], 1:] - pop_matrix[parents[i + 1], 1:])
                - pop_matrix[parents[i + 1], 1:]
            )

        self.pop_matrix[self.elitism_pop :] = new_pop_matrix

    # basic mutation (although no idea how effective)
    def mutation(self):
        reshaped_pop = np.reshape(self.pop_matrix[:, 1:], 2 * self.n_pop * self.N)
        mutate_locations = np.random.randint(
            2 * self.N * self.elitism_pop, 2 * self.N * self.n_pop, self.mutation_num
        )
        # mutate_locations = np.random.randint(0, 2 * self.N * self.n_pop, self.mutation_num)
        for location in mutate_locations:
            reshaped_pop[location] = (random.random()) * (
                self.x_bound[1] - self.x_bound[0]
            ) + self.x_bound[0]

        self.pop_matrix[:, 1:] = np.reshape(reshaped_pop, (self.n_pop, 2 * self.N))


# just for testing!

# pop_size_array = [17, 21, 33, 41, 61, 101, 151, 181]
# pop_size_array = [16, 20, 32, 40, 60, 100, 150, 180, 210]
pop_size = 200
# pop_size = 201
x_bound = [0, 2000]
y_bound = [0, 2000]
# n_turbines_array = [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# n_turbines_array = [26]
# n_turbines_array = [35, 40, 45]
n_turbines = 26
elitism_num = 10
crossover_rate = 0.8
mutation_rate = 0.35
# purge_pop_ratio = 0.5
grid_size = 40 * 5

fitness = np.zeros(400)
# for j in range(12):
#     # pop_size = pop_size_array[j]
#     n_turbines = n_turbines_array[j]

cost_value = np.zeros(pop_size)
power_value = np.zeros(pop_size)
penalty_value = np.zeros(pop_size)
a = time.time()

oldPop = Population(
    pop_size,
    x_bound,
    y_bound,
    n_turbines,
    elitism_num,
    crossover_rate,
    mutation_rate,
    grid_size,
)
oldPop.initialise_random()
oldPop.fitness_pop(cost_value, power_value)

avg_cost = cost_value.sum() / pop_size
avg_power = power_value.sum() / pop_size
avg_penalty = penalty_value.sum() / pop_size

# best_cost = avg_cost
# best_layout = oldPop.pop_matrix[0, 1:]

for i in range(400):
    # print("number of generation - ", i)
    newPop = Population(
        pop_size,
        x_bound,
        y_bound,
        n_turbines,
        elitism_num,
        crossover_rate,
        mutation_rate,
        grid_size,
    )
    newPop.elitism(oldPop.pop_matrix)
    newPop.crossover_greedy(oldPop.pop_matrix)
    newPop.mutation()
    newPop.fitness_pop(cost_value, power_value)
    # if i % 50 == 0:
    #     newPop.purge(purge_pop_ratio)
    # if best_cost < cost_value[0]:
    #     best_cost = cost_value[0]
    #     best_layout = newPop.pop_matrix[0, 1:]
    print(
        "{:0=3d} generation\nCost1: {}\nCost2: {}\nCost3: {}\nCost4: {}\nCost5: {}".format(
            i + 1,
            newPop.pop_matrix[0, 0],
            newPop.pop_matrix[1, 0],
            newPop.pop_matrix[2, 0],
            newPop.pop_matrix[3, 0],
            newPop.pop_matrix[4, 0],
        )
    )
    fitness[i] = newPop.pop_matrix[0, 0]
    oldPop = newPop

b = time.time()

x = newPop.pop_matrix[0, 1::2]
y = newPop.pop_matrix[0, 2::2]
# x = best_layout[::2]
# y = best_layout[1::2]
best_cost = cost_value[0]
best_power = power_value[0]
best_penalty = penalty_value[0]
print("avg_cost - ", avg_cost)
print("best_cost - ", best_cost)
print("avg_power - ", avg_power)
print("best_power - ", best_power)
print("avg_penalty - ", avg_penalty)
print("final_penalty - ", best_penalty)
generations = np.linspace(1, 400, 400)
# plt.plot(generations, fitness, 'r')
plt.show()
# print("{}, {}".format(n_turbines, b-a))
get_wake_plots(x, y, n_turbines, avg_cost, best_cost, b - a)
