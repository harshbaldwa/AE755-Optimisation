# not working
import random
import numpy as np
from ..common import initial_random_layout
from ..common import objective_function


class Population:
    def __init__(self, n_pop, x_bound, y_bound, N, crossover_prob, grid_size):
        self.n_pop = n_pop
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.pop_matrix = np.zeros((n_pop, 2 * N + 1))
        self.N = N
        self.crossover_pop = np.floor(crossover_prob * n_pop)
        self.grid_size = grid_size

    def initialise_random(self):
        for i in range(self.n_pop):
            self.pop_matrix[i, 1:] = initial_random_layout.random_layout(
                self.N, self.x_bound, self.y_bound, self.grid_size
            ).layout

    # make it compatible
    def fitness_pop(self, pop_matrix):
        for i in range(self.n_pop):
            pop_matrix[i, 0] = objective_function(pop_matrix[i, 1:])

        self.pop_matrix = self.pop_matrix[np.argsort(self.pop_matrix[:, 0])]

    # make beta different for all variables
    def crossover(self):
        new_pop_matrix = np.zeros((self.n_pop, 2 * self.N + 1))
        for i in range(self.n_pop):
            r = random.sample(range(self.crossover_pop), 3)
            beta = r[0] / self.crossover_pop
            new_pop_matrix[i, 1:] = (
                beta * self.pop_matrix[r[1], 1:]
                + (1 - beta) * self.pop_matrix[r[2], 1:]
            )

        self.new_pop_matrix = new_pop_matrix

    def mutation(self):
        pass


oldPop = Population(10, [0, 50], [0, 50], 23)
oldPop.initialise_random()
oldPop.fitness_pop()
for i in range(20):
    newPop = Population(10, [0, 50], [0, 50], 23)
    newPop.crossover()
    newPop.mutation()
    newPop.fitness_pop(newPop.new_pop_matrix)
    print("{}th generation - best cost: {}".format(i, newPop.pop_matrix[0].cost))
    oldPop = newPop
