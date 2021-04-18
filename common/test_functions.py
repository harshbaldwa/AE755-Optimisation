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

def rosenbrock(x, *args):
    x1 = x[0]
    x2 = x[1]

    r = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    return r

def ackley(x):
    x = np.array(x)
    a = 20.0
    b = 0.2
    c = 2*np.pi
    d = len(x)
    e = 1/d

    f = -a * np.exp(-b*np.sqrt((e*sum(x**2)))) - np.exp(e*sum(np.cos(c*x))) + a + np.exp(1)

    return f

def eggholder(x):
    x1 = x[0]
    x2 = x[1]

    penalty = 0
    if x1 > 512 or x2 >512 or x1<-512 or x2<-512:
        penalty = 10000

    term1 = -1*(x2+47) * np.sin(np.sqrt(np.abs(x2 + 0.5*x1+47)))
    term2 = -x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))

    return term1 + term2 + penalty

def holder_table(x):
    x1 = x[0]
    x2 = x[1]

    penalty = 0
    if x1 > 10 or x2 > 10 or x1<-10 or x2<-10:
        penalty = 10000


    f = -1*np.abs(np.sin(x1)*np.cos(x2)*np.exp(np.abs(1 - ((np.sqrt(x1**2 + x2**2))/np.pi))))
    # print(f, np.abs(1 - ((np.sqrt(x1**2 + x2**2)))))
    return f + penalty