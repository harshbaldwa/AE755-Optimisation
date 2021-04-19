#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.random import rand
# from ..common.cost import objective as obj
from ..common.test_functions import holder_table as obj
# from ..common.test_functions import himmel as obj

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


def give_direction(X,Y,step_size, E):

    X1 = X
    Y1 = Y
    new = X1 + step_size
    X1 = new
    o1 = obj([X1, Y1])

    X2 = X
    Y2 = Y
    new = Y2 + step_size
    Y2 = new
    o2 = obj([X2, Y2])


    X3 = X
    Y3 = Y
    new = X3 - step_size
    X3 = new
    o3 = obj([X3, Y3])


    X4 = X
    Y4 = Y
    new = Y4 - step_size
    Y4 = new
    o4 = obj([X4, Y4])

    min_obj = np.min([o1,o2,o3,o4])

    if(o1 == min_obj and o1<E):
        return X1, Y1

    if(o2 == min_obj and o2<E):
        return X2, Y2

    if(o3 == min_obj and o3<E):
        return X3, Y3

    if(o4 == min_obj and o4<E):
        return X4, Y4

    else:
        return X, Y


def compare(X,Y,X_old,Y_old):
    truth1 = X == X_old
    truth2 = Y == Y_old

    # print(truth1, truth2)

    if (truth1) or (truth2):
        return 1
    else:
        return 0



################### Pattern Search Inputs and loops starts here ################
n = 1

step_size = 10.0
min_step_size = 0.00001

####### for lool for checking constrint till satisfection ######
X = np.random.random()*10 - 5
Y = np.random.random()*10 - 5

E = obj([X, Y])

X_old = X
Y_old = Y

################ Generate random selection order ##############
order = random_selection_order(n)

while(step_size>min_step_size):
    # print(step_size)
    k = 0.0
    while(True):
        k += 1
        X_old = X
        Y_old = Y
        X, Y = give_direction(X,Y,step_size, E)

        E = obj([X, Y])

        if X == X_old and Y == Y_old:
            break;

    step_size /= 2

# print(365*24*aep(positions, windspeed_array, theta_array, wind_prob, alpha, 0.5*diameter, boundary_limits)[0])
print(X, Y, E)
