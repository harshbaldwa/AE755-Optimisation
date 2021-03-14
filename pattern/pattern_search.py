#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.random import rand
from cost import obj
from popping import relocate_turbines



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

    X1 = X
    Y1 = Y
    new = X1[turbine_num] + step_size
    if(new < X_limit):
        ##### constraint #####
        X1[turbine_num] = new
        o1 = obj(get_pos(X1,Y1))
        print(obj(get_pos(X1,Y1)), "objective")
    else:
        o1 = 0

    X2 = X
    Y2 = Y
    new = Y2[turbine_num] + step_size
    if(new < Y_limit):
        ##### constraint #####
        Y2[turbine_num] = new
        o2 = obj(get_pos(X2,Y2))
    else:
        o2 = 0

    X3 = X
    Y3 = Y
    new = X3[turbine_num] - step_size
    if(new > 0):
        ##### constraint #####
        X3[turbine_num] = new
        o3 = obj(get_pos(X3,Y3))
    else:
        o3 = 0

    X4 = X
    Y4 = Y
    new = Y4[turbine_num] - step_size
    if(new > 0):
        ##### constraint #####
        Y4[turbine_num] = new
        o4 = obj(get_pos(X4,Y4))
    else:
        o4 = 0

    max_obj = np.max([o1,o2,o3,o4])
    if(o1 == max_obj and o1>E):
        return X1,Y1
    if(o2 == max_obj and o2>E):
        return X2,Y2
    if(o3 == max_obj and o3>E):
        return X3,Y3
    if(o3 == max_obj and o4>E):
        return X4,Y4
    else:
        return X, Y

def compare(X,Y,X_old,Y_old):
    comparison1 = (X == X_old)
    comparison2 = (Y == Y_old)
    equal1 = comparison1.all()
    equal2 = comparison2.all()
    result = equal1 + equal2
    if (result==2):
        return 1
    else:
        return 0



################### Pattern Search Inputs and loops starts here ################
n = 5
X_limit = 4000
Y_limit = 3500
step_size = 500
min_step_size = 1
diameter = 82

####### for lool for checking constrint till satisfection ######
X = random_cordi(n, X_limit)
Y = random_cordi(n, Y_limit)

################# Evaluate Objective Function ##################
positions = np.zeros(2*len(X))
positions[::2] = X
positions[1::2] = Y


# E = objective(X, Y)
E = obj(positions)

X_old = X.copy()
Y_old = Y.copy()

################ Generate random selection order ##############
order = random_selection_order(n)
# print(order)

while(step_size>min_step_size):
    print(step_size)
    k = 0.0
    while(compare(X,Y,X_old,Y_old)==0):
        k += 1
        ################ Generate random selection order ##############
        order = random_selection_order(n)
        # print(order)

        ############# for every turbine & For every direction ##########
        # print(X,Y)
        for i in range(n):
            X_new,Y_new = give_direction(X,Y,order[i],step_size,X_limit,Y_limit, E)
            # print(i, n)
            X = X_new
            Y = Y_new

            positions[::2] = X
            positions[1::2] = Y

            # E = objective(X,Y)
            E = obj(positions)

        plt.scatter(X, Y, label=str(k))
        if k % 5 == 0 :
            plt.legend()
            plt.show()
            # plt.lf()

        # plt.show()
    ######## Call popping algorithm ########
    print("Test")
    X, Y = relocate_turbines(X, Y, boundary_limits=[[0 , X_limit], [0, Y_limit]],
                            n_weak_turbines=5, n_reloc_attempts=10, diameter=diameter)

    step_size /= 2


