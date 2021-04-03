import numpy as np
from ..common.wake_model import aep
from ..common.cost import obj


def relocate_turbines(x, y, boundary_limits=[[0.0, 1.0], [0.0, 1.0]], n_weak_turbines=5, n_reloc_attempts=15, diameter=82):
    power_produced = np.zeros_like(x)

    Z_H = 60  # Hub height of rotor in m
    Z_0 = 0.3  # Hub height of rotor in m
    alpha = 0.5 / (np.log(Z_H / Z_0))

    positions = np.zeros(2*len(x))
    positions[::2] = x
    positions[1::2] = y

    for i in range(len(x)):
        position = np.array([x[i], y[i]])
        power_produced[i], penalty = aep(position, [12, 0], alpha=alpha, rr=0.5*diameter, boundary_limits=boundary_limits)


    # power_produced = get_power(x, y) # get power produced by the turbines

    # boundary limits
    xlimits = boundary_limits[0]
    xlen = xlimits[1] - xlimits[0]
    ylimits = boundary_limits[1]
    ylen = ylimits[1] - ylimits[0]

    for i in range(n_weak_turbines):

        idx = np.argmin(power_produced) #identify the weakest turbine

        # E = 0 # objective function call
        E = obj(positions)

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

            E_relocated = obj(pos_new)

            if E_relocated < E:
                x = x_pos
                y = y_pos
                break;

            # else:
            #     x_pos = x.copy()
            #     y_pos = y.copy()

        for i in range(len(x)): # get power produced by the turbines
            power_produced[i], penalty = aep(np.array([x[i], y[i]]), [12, 0], alpha=alpha, rr=0.5*diameter)

    return x, y


