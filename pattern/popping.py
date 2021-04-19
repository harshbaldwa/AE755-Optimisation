import numpy as np
from ..common.wake_model import aep
# from ..common.cost import objective as obj
from ..common.mosetti_cost import objective as obj



def relocate_turbines(x, y, n_weak_turbines, n_reloc_attempts, boundary_limits,
                      diameter, Z_H, Z_0, windspeed_array, theta_array, wind_prob):

    power_produced = np.zeros_like(x)
    alpha = 0.5 / (np.log(Z_H / Z_0))

    positions = np.zeros(2*len(x))
    positions[::2] = x
    positions[1::2] = y

    for i in range(len(x)):
        position = np.array([x[i], y[i]])
        power_produced[i], penalty = aep(position, windspeed_array,
                                         theta_array, wind_prob, alpha,
                                         0.5*diameter, boundary_limits)

    # power_produced = get_power(x, y) # get power produced by the turbines

    # boundary limits
    xlimits = boundary_limits[0]
    xlen = xlimits[1] - xlimits[0]
    ylimits = boundary_limits[1]
    ylen = ylimits[1] - ylimits[0]

    for i in range(n_weak_turbines):

        idx = np.argmin(power_produced) #identify the weakest turbine

        # E = 0 # objective function call
        E = obj(positions, boundary_limits, diameter, Z_H,
                Z_0, windspeed_array, theta_array, wind_prob)

        x_pos = x.copy()
        y_pos = y.copy()

        for j in range(n_reloc_attempts):
            # get relocation points
            x_new = xlen * np.random.random() + xlimits[0]
            y_new = ylen * np.random.random() + ylimits[0]

            x_pos[idx] = x_new
            y_pos[idx] = y_new

            # E_relocated = 0 # objective function call
            pos_new = np.zeros(2*len(x_pos))
            pos_new[::2] = x_pos
            pos_new[1::2] = y_pos

            E_relocated = obj(pos_new, boundary_limits, diameter,
                              Z_H, Z_0, windspeed_array, theta_array, wind_prob)

            if E_relocated < E:
                x = x_pos
                y = y_pos
                break;

        for i in range(len(x)): # get power produced by the turbines
            power_produced[i], penalty = aep(position, windspeed_array, theta_array,
                                             wind_prob, alpha, 0.5*diameter, boundary_limits)

    return x, y


