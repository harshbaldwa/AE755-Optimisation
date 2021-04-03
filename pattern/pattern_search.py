#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.random import rand
# from ..common.cost import obj
from ..common.mosetti_cost import mosetti as obj
from ..common.wake_visualization import get_wake_plots
from .popping import relocate_turbines
from ..common.layout import Layout


######################### All function definations ####################
def random_cordi(n,limit):
    coordi = np.random.uniform(0,limit,n)
    return coordi

def objective(X, Y):
    a = np.sum(X) + np.sum(Y)
    return(a)

def random_selection_order(n):
    num_list = random.sample(range(n), n)
    random.shuffle(num_list)
    return num_list

def get_pos(X, Y):
    positions = np.zeros(2*len(X))
    positions[::2] = X
    positions[1::2] = Y

    return positions

def give_direction(X,Y,turbine_num,step_size, X_limit, Y_limit, E):

    X1 = X.copy()
    Y1 = Y.copy()
    new = X1[turbine_num] + step_size
    X1[turbine_num] = new
    o1 = obj(get_pos(X1,Y1))


    X2 = X.copy()
    Y2 = Y.copy()
    new = Y2[turbine_num] + step_size
    Y2[turbine_num] = new
    o2 = obj(get_pos(X2,Y2))


    X3 = X.copy()
    Y3 = Y.copy()
    new = X3[turbine_num] - step_size
    X3[turbine_num] = new
    o3 = obj(get_pos(X3,Y3))


    X4 = X.copy()
    Y4 = Y.copy()
    new = Y4[turbine_num] - step_size
    Y4[turbine_num] = new
    o4 = obj(get_pos(X4,Y4))

    min_obj = np.min([o1,o2,o3,o4])
    # print("Objective values", o1, o2, o3, o4, E)

    if(o1 == min_obj and o1<E):
        # print("1")
        return X1,Y1
    if(o2 == min_obj and o2<E):
        # print("2")
        return X2,Y2
    if(o3 == min_obj and o3<E):
        # print("3")
        return X3,Y3
    if(o4 == min_obj and o4<E):
        # print("4")
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
n = 20
X_limit = 2000
Y_limit = 2000
step_size = 250
min_step_size = 1
diameter = 40

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

E = obj(positions)
# print(E)
X_old = X.copy()
Y_old = Y.copy()

################ Generate random selection order ##############
order = random_selection_order(n)

# plt.scatter(X, Y, label="Initial")

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
            # plt.scatter(X, Y, label='initial')
            # print("initial E, order", E, order[i])
            x_store, y_store = X[i], Y[i]
            X, Y = give_direction(X,Y,order[i],step_size,X_limit,Y_limit, E)
            # get_wake_plots(X, Y)
            # plt.scatter([x_store], [y_store], marker='*')

            # plt.scatter(X, Y, "updated")
            positions[::2] = X
            positions[1::2] = Y

            E = obj(positions)
            # print("updated E", E)
            # print("\n")
            # plt.clf()

        if compare(X,Y,X_old,Y_old) == 0:
            break;

    ######## Call popping algorithm ########

    X, Y = relocate_turbines(X, Y, boundary_limits=[[0 , X_limit], [0, Y_limit]],
                            n_weak_turbines=5, n_reloc_attempts=10, diameter=diameter)

    step_size /= 2

# print(X, Y)
# print(E)
# plt.scatter(X, Y, label="Final")
# plt.ylim(0, Y_limit)
# plt.xlim(0, X_limit)
# plt.legend()
# plt.show()
get_wake_plots(X, Y)


