import numpy as np

def relocate_turbines(x, y, boundary_limits=[[0.0, 1.0], [0.0, 1.0]], n_weak_turbines=5, n_reloc_attempts=15):
    power_produced = get_power(x, y) # get power produced by the turbines

    # boundary limits
    xlimits = boundary_limits[0]
    xlen = xlimits[1] - xlimits[0]
    ylimits = boundary_limits[1]
    ylen = ylimits[1] - ylimits[0]

    for i in range(n_weak_turbines):

        idx = np.argmin(power_produced) #identify the weakest turbine

        E = 0 # objective function call

        x_pos = x.copy()
        y_pos = y.copy()

        for j in range(n_reloc_attempts):
            # get relocation points
            x_new = xlen * np.random.random() + xlimits[0]
            y_new = ylen * np.random.random() + ylimits[0]

            x_pos[idx] = x_new
            y_pos[idx] = y_new

            E_relocated = 0 # objective function call

            if E_relocated > E:
                x = x_pos
                y = y_pos
                break;

            else:
                x_pos = x.copy()
                y_pos = y.copy()

        power_produced = get_power(x, y) # get power produced by the turbines

    return x, y


