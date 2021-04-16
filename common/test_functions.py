#!/usr/bin/env python3
import math
import numpy as np
from .wake_model import aep
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