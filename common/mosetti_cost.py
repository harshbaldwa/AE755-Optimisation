import math
from .wake_model import aep

def mosetti(x, x_bound=[0, 2000], y_bound=[0, 2000]):
    n = len(x)/2
    cost = n*(2/3 + 1/3*math.exp(-0.00174*n**2))
    alpha = 0.5/math.log(60/0.3)
    P, penalty = aep(x, [12, 0], alpha, 40, [x_bound, y_bound])

    # print(cost*365*24/P, 1e-6*penalty)
    return cost*365*24/P + 1e-6*penalty
