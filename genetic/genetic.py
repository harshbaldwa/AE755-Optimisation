import numpy as np

class Population:
	def __init__(self, n_pop, x_bound, y_bound, N):
		self.n_pop = n_pop
		self.x_bound = x_bound
		self.y_bound = y_bound
		self.pop_matrix = np.zeros(n_pop, 2*N)
		self.elite_var = np.floor(0.1*n_pop)

	def initialise_random(self):
		pass

	def fitness_pop(self):
		pass

	def elite(self, oldPop):
		self.pop_matrix[:]

	# only for new layouts
	def mutation(self):
		pass

	def crossover(self):
		pass

class Layout:
	def __init__(self, X, Y, N):
		self.N = N
		layout = np.zeros(2*N)
		layout[::2] = X
		layout[1::2] = Y
		self.layout = layout

	def calc_cost(self):
		self.cost = function_call(self.layout)


oldPop = Population(10, [0, 50], [0, 50], 23)
oldPop.initialise_random()
oldPop.fitness_pop()
for i in range(10):
	newPop = Population(10, [0, 50], [0, 50], 23)
	newPop.elite()
	newPop.crossover()
	newPop.mutation()
	newPop.fitness_pop()
	print("{}th generation - best cost: {}".format(i, newPop.pop_matrix[0].cost))
	oldPop = newPop
