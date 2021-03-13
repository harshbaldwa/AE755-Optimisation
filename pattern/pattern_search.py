#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random 
from numpy.random import rand



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

def give_direction(X,Y,turbine_num,step_size, X_limit, Y_limit, E):

    X1 = X
    Y1 = Y
    new = X1[turbine_num] + step_size
    if(new < X_limit):
        ##### constraint #####
        X1[turbine_num] = new
        o1 = objective(X1,Y1)
    else:
        o1 = 0
    
    X2 = X
    Y2 = Y
    new = Y2[turbine_num] + step_size
    if(new < Y_limit):
        ##### constraint #####
        Y2[turbine_num] = new
        o2 = objective(X2,Y2)
    else:
        o2 = 0
    
    X3 = X
    Y3 = Y
    new = X3[turbine_num] - step_size
    if(new > 0):
        ##### constraint #####
        X3[turbine_num] = new
        o3 = objective(X3,Y3)
    else:
        o3 = 0
    
    X4 = X
    Y4 = Y
    new = Y4[turbine_num] - step_size
    if(new > 0):
        ##### constraint #####
        Y4[turbine_num] = new
        o4 = objective(X4,Y4)
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
n = 20
X_limit = 4000
Y_limit = 3500
step_size = 10
min_step_size = 1

####### for lool for checking constrint till satisfection ######
X = random_cordi(n, X_limit)
Y = random_cordi(n, Y_limit)

################# Evaluate Objective Function ##################
E = objective(X, Y)

X_old = X
Y_old = Y

################ Generate random selection order ##############
order = random_selection_order(n)
# print(order)

while(step_size>min_step_size):
    
    while(compare(X,Y,X_old,Y_old)==0):
        
        ################ Generate random selection order ##############
        order = random_selection_order(n)
        # print(order)
        
        ############# for every turbine & For every direction ##########
        print(X,Y)
        for i in range(n):
            X_new,Y_new = give_direction(X,Y,order[i],step_size,X_limit,Y_limit, E)
            X = X_new
            Y = Y_new
            E = objective(X,Y)
    
    ######## Call popping algorithm ########
    
        
    step_size /= 2
    

