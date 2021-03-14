import math
import numpy as np
from .wake_model import aep
from .constraint_check import penalty_function


def obj(layout, x_bound=[0, 4000], y_bound=[0, 3500]):  # OBJECTIVE FUNCTION THAT WE'RE TRYING TO MINIMISE

    SP = 0.02  # selling price in $/kWhr
    r_i = 0.02  # interest rate - inflation rate
    T = 20  # lifespan of farm in years
    N = int(0.5 * len(layout))  # number of turbines
    P_rated = 1.5  # Rated power of a single turbine in kW
    D = 82  # Rotor Diameter in m
    Z_H = 60  # Hub height of rotor in m
    Z_0 = 0.3  # Hub height of rotor in m
    A = 0.25 * np.pi * D * D

    alpha = 0.5 / (math.log(Z_H / Z_0))
    AEP = aep(layout, [1, 0], alpha, D / 2)

    L_coll = 0
    for i in range(N):
        L_coll += np.sqrt(
            layout[2 * i] ** 2 + layout[2 * i + 1]**2
        )  # Assuming turbine-wise coordinates

    C_RNA = 1170 * P_rated * N

    C_tower = 1.5 * 0.016 * (D ** 2.8) * ((Z_H / D) ** 1.7) * ((P_rated / A) ** 0.6) * N

    C_base = 22.5 * (60 + 7.78) * N

    C_SS = C_tower + C_base

    C_elec = L_coll * (0.0105 * 10000 + 95) + np.ceil(
        0.603 * P_rated * N / 100000 + 0.464
    ) * (
        (0.00168 * 100000 + 1380) * 1000
        + 0.668 * 100000
        + 36000
        + 1.035 * (1000 * (P_rated) ** 0.751) * N
    )

    C_decom = P_rated * N * 55 * 13

    a = (1 - 1 / ((1 + r_i) ** T)) / r_i

    # C_inv = C_RNA + C_SS + C_elec + C_decom
    C_inv = C_RNA + C_SS + C_decom
    C_OM = 0.25 * C_inv / a

    LPC = C_inv / (a * AEP) + C_OM / AEP
    penalty = penalty_function(layout, boundary_limits = [x_bound, y_bound], diameter=D)
    objective = (LPC - SP) * AEP + penalty
    # print(penalty, "penalty")
    # print(objective, "objective")

    # return (LPC - SP) * AEP + penalty_function(layout, boundary_limits = [x_bound, y_bound], diameter=D)
    return objective