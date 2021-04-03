import numpy as np
import matplotlib.pyplot as plt

def get_turbine_lines(x, y, diameter, height, z_0, wind):
    normal = np.array([-wind[1]/(wind[0]+np.finfo(float).eps), 1])
    normal = normal/np.linalg.norm(normal)
    R = diameter/2

    Z_H = height
    Z_0 = z_0
    alpha = 0.5 / (np.log(Z_H / Z_0))

    for i in range(len(x)):
        turbine_pos = np.array([x[i], y[i]])

        pos1 = turbine_pos + np.array([0, R])
        pos2 = turbine_pos - np.array([0, R])

        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='red')

        dist = 12.2*R
        rl = alpha*dist + R

        pos4 = turbine_pos + np.array([dist, rl])
        pos3 = turbine_pos - np.array([0, rl]) + np.array([dist, 0])

        x_pos = [pos1[0], pos2[0], pos3[0], pos4[0]]
        y_pos = [pos1[1], pos2[1], pos3[1], pos4[1]]

        plt.fill(x_pos, y_pos, facecolor='lightsalmon', edgecolor='orangered', alpha=0.25)

def get_turbine_circles(x, y, diameter):
    R = 2.5 * diameter

    for i in range(len(x)):

        x_range = np.linspace(x[i] - 0.99*R, x[i] + 0.99*R, 100)
        y_range = y[i] + np.sqrt((R ** 2) - (x_range - x[i])**2)
        y_range_2 = y[i] - np.sqrt((R ** 2) - (x_range - x[i])**2)

        y_range = np.concatenate((y_range, y_range_2[::-1]))
        x_range = np.concatenate((x_range, x_range[::-1]))


        plt.fill(x_range, y_range, facecolor='lightsteelblue', edgecolor='royalblue', alpha=0.2)



def get_wake_plots(x, y, bounds, diameter, height, z_0, wind_velocity):
    """
    x, y: 1D arays are turbines coordinates
    wind: 1x2 array of x and y components of the wind.
    """
    wind = [wind_velocity, 0]
    fig = plt.figure()
    get_turbine_lines(x, y, diameter, height, z_0, wind)
    get_turbine_circles(x, y, diameter)
    plt.scatter(x, y, color='black')
    plt.xlim(bounds[0, 0], bounds[0, 1])
    plt.ylim(bounds[1, 0], bounds[1, 1])
    plt.gca().set_aspect('equal')
    plt.show()


