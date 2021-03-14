#!/usr/bin/env python3
import math
import numpy as np
from .wake_model import aep
from .constraint_check import penalty_function

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

def mosetti(x, *args):
    n = len(x)/2
    cost = n*(2/3 + 1/3*math.exp(-0.00174*n**2))
    alpha = 0.5/math.log(60/0.3)
    P = aep(x, [12, 0], alpha, 20)/(365*24*3600)

    return cost/P + 1e-10*penalty_function(x)


# if __name__=="__main__":
#     print(Bohachevsky([3,3]))
