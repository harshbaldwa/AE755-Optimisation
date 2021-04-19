import time
import numpy as np
from .mosetti_cost import objective as simple
from .mosetti_cost_gen import objective as gen

def init_random(pop, n_pop, bounds, N):
    for i in range(n_pop):
        pop[i, ::2] = bounds[0, 0] + np.random.random(N) * (
            bounds[0, 1] - bounds[0, 0]
        )
        pop[i, 1::2] = bounds[1, 0] + np.random.random(N) * (
            bounds[1, 1] - bounds[1, 0]
        )

bounds = np.array([[0, 4000], [0, 3500]])
n_pop = 3
N = 47
pop = np.zeros((n_pop, 2*N))
init_random(pop, n_pop, bounds, N)
windsp = np.array([12])
theta = np.array([0])
prob = np.array([1])

cs = np.zeros(n_pop)

a = time.time()
for i in range(n_pop):
    cs[i] = simple(pop[i], bounds, 82, 80, 0.3, windsp, theta, prob)
b=time.time()
cg = gen(pop, bounds, 82, 80, 0.3, windsp, theta, prob)
c = time.time()

print(cs-cg)
# print(cg)

# print((b-a)/(c-b))

