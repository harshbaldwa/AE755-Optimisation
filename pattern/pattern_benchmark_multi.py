import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.random import rand
# from ..common.cost import objective as obj
from ..common.test_functions import ackley as obj
# from ..common.mosetti_cost import objective as obj
from ..common.wake_visualization import get_wake_plots
from .popping import relocate_turbines
from ..common.layout import Layout
from ..common.wake_model import aep


######################### All function definations ####################
def random_cordi(n,limit):
    coordi = np.random.uniform(0,limit,n)
    return coordi


def random_selection_order(n):
    num_list = random.sample(range(n), n)
    random.shuffle(num_list)
    return num_list


def get_pos(X, Y):
    positions = np.zeros(2)
    positions[::2] = X
    positions[1::2] = Y

    return positions


def give_direction(X,step_size, E, idx):

    X1 = X.copy()
    new = X1[idx] + step_size
    X1[idx] = new
    o1 = obj(X1)

    X2 = X.copy()
    new = X2[idx] - step_size
    X2[idx] = new
    o2 = obj(X2)

    min_obj = np.min([o1,o2])

    if(o1 == min_obj and o1<E):
        return X1

    if(o2 == min_obj and o2<E):
        return X2

    else:
        return X


def compare(X, X_old):
    truth1 = X == X_old

    if (False in truth1):
        return 1
    else:
        return 0



################### Pattern Search Inputs and loops starts here ################
n = 6

step_size = 10
min_step_size = 0.000001

####### for lool for checking constrint till satisfection ######
X = np.random.random(n)*40 - 20

E = obj(X)

X_old = X.copy()

################ Generate random selection order ##############
order = random_selection_order(n)

while(step_size>min_step_size):
    print(step_size)
    while(True):
        X_old = X.copy()
        for i in range(n):
            X = give_direction(X,step_size, E, order[i])
            E = obj(X)

        if compare(X, X_old):
            break;

    step_size /= 2

print(X)
print(E)