#!/usr/bin/env python3
import math
import numpy as np
# from .wake_model import aep
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

def ackely(x, *args):
    a = 20
    b = 0.2
    c = 2*np.pi 
    d = len(x)

    t1 = -a * np.exp(-b* np.sqrt(np.sum(x**2)/d))
    t2 = -np.exp(np.sum(np.cos(c*x))/d)
    pen = np.abs(x)>32
    penalty = 1e8*np.sum(pen)
    return t1+t2+a+np.exp(1)+penalty

def eggholder(x, *args):
    '''
    expected best cost is -959.6 @ (512, 404.23)

    '''
    x1 = x[0]
    x2 = x[1]

    pen = np.abs(x)>512
    penalty = np.sum(pen)*1e20

    term1 = -(x2+47) * np.sin(np.sqrt(np.abs(x2 + 0.5*x1+47)))
    term2 = -x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))

    return term1 + term2 + penalty

def rosenbrock(x, *args):
    x1 = x[0]
    x2 = x[1]

    r = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    return r

def holder_table(x, *args):
    '''
    expected: -19.2085 @ (+-8.05502, +-9.66499)
    '''
    x1 = x[0]
    x2 = x[1]

    pen = np.abs(x)>10
    penalty = np.sum(pen)*1e10


    f = -1*np.abs(np.sin(x1)*np.cos(x2)*np.exp(np.abs(1 - ((np.sqrt(x1**2 + x2**2))/np.pi))))
    return f + penalty
