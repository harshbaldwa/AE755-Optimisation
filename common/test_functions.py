#!/usr/bin/env python3
import math
import numpy as np
from .wake_model import aep, penalty_function
# from .constraint_check import penalty_function

def Bohachevsky(x, *args):
    x1 = x[0]
    x2 = x[1]
    r = x1**2 + x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7
    return r

def himmel(x, *args):
    x1 = x[0]
    x2 = x[1]

    r = (x1**2 + x2 -11)**2 + (x1 +x2**2 -7)**2
    return r

def himmel_partial(x, *args):
    x1 = x[0]
    x2 = x[1]

    r = (x1**2 + x2 -11)**2 + (x1 +x2**2 -7)**2 + 10*np.abs((x1-3)*(x2-2))
    return r

def ackley(x, *args):
    n = len(x)
    a = 20
    b = 0.2
    c = 2*np.pi
    
    value = -a*np.exp(-b*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(c*x))/n) + a + np.exp(1)
    return value

def eggholder(x, *args):
    x1 = x[0]
    x2 = x[1]

    value = -(x2+47)*np.sin(np.sqrt(np.abs(x2+x1/2+47))) - x1*np.sin(np.sqrt(np.abs(x1 - (x2+47))))
    penalty = 0
    if abs(x1) > 512:
        penalty += 1000*(abs(x1) - 512)**2
    if abs(x2) > 512:
        penalty += 1000*(abs(x2) - 512)**2
    return value + penalty

def holder_table(x, *args):
    x1 = x[0]
    x2 = x[1]

    f = -1*np.abs(np.sin(x1)*np.cos(x2)*np.exp(np.abs(1 - ((np.sqrt(x1**2 + x2**2))/np.pi))))

    penalty = 0
    if x1 > 10 or x2 > 10 or x1<-10 or x2<-10:
        penalty = 10000
    
    return f + penalty
