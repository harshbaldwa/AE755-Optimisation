import numpy as np
from wake_model import aep
# from constraint_check import penalty_function

def obj(layout, x_bound=[0, 4000], y_bound=[0, 3500]):  # OBJECTIVE FUNCTION THAT WE'RE TRYING TO MINIMISE


    SP = 0.02  # selling price in $/kWhr [0.02]
    r_i = 0.02  # interest rate - inflation rate [0.02]
    T = 20  # lifespan of farm in years [20]
    P_rated = 1.5  # Rated power of a single turbine in kW [1.5]
    N = int(0.5 * len(layout))  # number of turbines
    D = 82  # Rotor Diameter in m [82]
    Z_H = 80  # Hub height of rotor in m [80]
    Z_0 = 0.3  # Surface roughness of terrain
    A = 0.25 * np.pi * D * D
    L_trans = 10000 # Length of transmission cable - running from farm to distribution center - in m [10e3]
    V_M = 10000 # Voltage rating of Medium Voltage AC (MVAC) lines used for collection [10e3]
    V_H = 100000 # Voltage rating of High Voltage AC (HVAC) lines used for transmission [100e3]
    
    L_coll = 0
    for i in range(N):
        L_coll += np.sqrt(
            layout[2 * i] ** 2 + layout[2 * i + 1]**2
        )  # Assuming turbine-wise coordinates, calculating length of collection cables - cables running from each turbine to collection center at (0,0)

    alpha = 0.5 / (np.log(Z_H / Z_0))
    AEP, penalty = aep(layout=layout, w=[12, 0], alpha=alpha, rr=D/2, boundary_limits=[x_bound, y_bound])
    AEP = AEP * 365 * 24
    # penalty = penalty_function(layout, boundary_limits = [x_bound, y_bound], diameter=D)

    ### START MULTILINE COMMENT, use commented lines if any of the parameters change from default values, indicated by []

    # C_RNA = 1170 * P_rated * N

    # C_tower = 1.5 * 0.016 * (D ** 2.8) * ((Z_H / D) ** 1.7) * ((P_rated * 1000 / A) ** 0.6) * N

    # C_base = 22.5 * (60 + 7.78) * N

    # C_SS = C_tower + C_base

    # C_elec = L_coll * (0.0105 * V_M + 95) + np.ceil(
    #     0.603 * P_rated * N / V_H + 0.464
    # ) * (
    #     (0.00168 * V_H + 1380) * L_trans
    #     + 0.668 * V_H
    #     + 36000
    #     + 1.035 * (1000 * P_rated * N) ** 0.751
    # )

    # C_decom = P_rated * N * 55 * 13

    # a = (1 - 1 / ((1 + r_i) ** T)) / r_i

    # C_inv = C_RNA + C_SS + C_elec + C_decom
    # C_OM = 0.25 * C_inv / a

    # LPC = C_inv / (a * AEP) + C_OM / AEP
    # # print(C_inv, C_OM, a)
    # objective = (LPC - SP) * AEP + penalty

    ### STOP MULTILINE COMMENT

    x_min = np.min(layout[::2])
    x_max = np.max(layout[::2])
    y_min = np.min(layout[1::2])
    y_max = np.max(layout[1::2])
    area = (y_max - y_min)*(x_max - x_min)
    land_cost = area*40878

    objective = 1191241.17052 + 15.28918*L_coll + 521.550195949*N + 19.210518346*(N**0.751) - 0.02*AEP + 1e-4*land_cost + 1e4*penalty

    # print(penalty, "penalty")
    # print(objective, "objective")

    # return (LPC - SP) * AEP + penalty_function(layout, boundary_limits = [x_bound, y_bound], diameter=D)
    # print(penalty)
    return objective