#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.random import rand
from ..common.cost import objective as obj
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
    positions = np.zeros(2*len(X))
    positions[::2] = X
    positions[1::2] = Y

    return positions


def give_direction(X,Y,turbine_num,step_size, X_limit, Y_limit, E, boundary_limits,
                   diameter, Z_H, Z_0, windspeed_array, theta_array, wind_prob):

    X1 = X.copy()
    Y1 = Y.copy()
    new = X1[turbine_num] + step_size
    X1[turbine_num] = new
    o1 = obj(get_pos(X1,Y1), boundary_limits, diameter,
             Z_H, Z_0, windspeed_array, theta_array, wind_prob)

    X2 = X.copy()
    Y2 = Y.copy()
    new = Y2[turbine_num] + step_size
    Y2[turbine_num] = new
    o2 = obj(get_pos(X2,Y2), boundary_limits, diameter,
             Z_H, Z_0, windspeed_array, theta_array, wind_prob)


    X3 = X.copy()
    Y3 = Y.copy()
    new = X3[turbine_num] - step_size
    X3[turbine_num] = new
    o3 = obj(get_pos(X3,Y3), boundary_limits, diameter,
             Z_H, Z_0, windspeed_array, theta_array, wind_prob)


    X4 = X.copy()
    Y4 = Y.copy()
    new = Y4[turbine_num] - step_size
    Y4[turbine_num] = new
    o4 = obj(get_pos(X4,Y4), boundary_limits, diameter,
             Z_H, Z_0, windspeed_array, theta_array, wind_prob)

    min_obj = np.min([o1,o2,o3,o4])

    if(o1 == min_obj and o1<E):
        return X1,Y1

    if(o2 == min_obj and o2<E):
        return X2,Y2

    if(o3 == min_obj and o3<E):
        return X3,Y3

    if(o4 == min_obj and o4<E):
        return X4,Y4

    else:
        return X, Y


def compare(X,Y,X_old,Y_old):
    truth1 = X == X_old
    truth2 = Y == Y_old

    if (False in truth1) or (False in truth2):
        return 1
    else:
        return 0



################### Pattern Search Inputs and loops starts here ################
n = 33

# Setup - Mossetti
# X_limit = 2000
# Y_limit = 2000
# boundary_limits = [[0, X_limit], [0, Y_limit]]
# step_size = 250
# min_step_size = 1
# diameter = 82
# Z_H = 60
# Z_0 = 0.3

# Setup - Nani Sindhodi
X_limit = 4000
Y_limit = 3500
boundary_limits = np.array([[0, X_limit], [0, Y_limit]])
step_size = 500
min_step_size = 1
diameter = 82
Z_H = 80
Z_0 = 0.3

alpha = 0.5 / (np.log(Z_H / Z_0))

# Wind Setup
windspeed_array = np.array([12])
theta_array = np.array([0.0])
wind_prob = np.array([1])


####### for lool for checking constrint till satisfection ######
X = random_cordi(n, X_limit)
Y = random_cordi(n, Y_limit)

# X = np.array([0, 100, 200, 300, 400, 500, 600])
# Y = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000])

# X = np.array([0, 50, 100])
# Y = np.array([1000, 1000, 1000])

################# Evaluate Objective Function ##################
positions = np.zeros(2*len(X))
positions[::2] = X
positions[1::2] = Y

E = obj(positions, boundary_limits, diameter, Z_H, Z_0, windspeed_array, theta_array, wind_prob)
X_old = X.copy()
Y_old = Y.copy()

################ Generate random selection order ##############
order = random_selection_order(n)

while(step_size>min_step_size):
    print(step_size)
    k = 0.0
    while(True):
        k += 1
        ################ Generate random selection order ##############
        order = random_selection_order(n)
        ############# for every turbine & For every direction ##########
        X_old = X.copy()
        Y_old = Y.copy()

        for i in range(n):
            x_store, y_store = X[i], Y[i]

            X, Y = give_direction(X,Y,order[i],step_size,X_limit,Y_limit, E,
                                  boundary_limits, diameter, Z_H, Z_0,
                                  windspeed_array, theta_array, wind_prob)

            positions[::2] = X
            positions[1::2] = Y

            E = obj(positions, boundary_limits, diameter, Z_H, Z_0, windspeed_array, theta_array, wind_prob)

        if compare(X,Y,X_old,Y_old) == 0:
            break;

    ######## Call popping algorithm ########
    n_pop = 5
    n_relocate = 10

    X, Y = relocate_turbines(X, Y, n_pop, n_relocate,
                             boundary_limits, diameter, Z_H, Z_0,
                             windspeed_array, theta_array, wind_prob)

    step_size /= 2

# print(365*24*aep(positions, windspeed_array, theta_array, wind_prob, alpha, 0.5*diameter, boundary_limits)[0])


algo_data = [
    "Pattern",
    "n_pop: {}\nn_relocate: {}\n".format(
        n_pop, n_relocate
    ),
    "n_turb: {}\ndiameter: {}\nheight: {}\ncost_model: {}\nprofit: ${:.2f}M".format(
        n, diameter, Z_H, 'tejas', E / 1e6
    ),
    "pattern_{}".format(n),
]

get_wake_plots(
    X,
    Y,
    boundary_limits,
    diameter,
    Z_H,
    Z_0,
    windspeed_array,
    theta_array,
    wind_prob,
    algo_data
)