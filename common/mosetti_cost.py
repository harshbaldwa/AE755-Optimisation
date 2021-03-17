import math
from .wake_model import aep

def mosetti(x, x_bound, y_bound):
    n = len(x)/2
    cost = n*(2/3 + 1/3*math.exp(-0.00174*n**2))
    alpha = 0.5/math.log(60/0.3)
    P, penalty = aep(x, [12, 0], alpha, 20, [x_bound, y_bound])

    return cost*365*24/P + 1e-10*penalty
