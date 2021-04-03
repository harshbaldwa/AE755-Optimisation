import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def get_turbine_lines(x, y, wind):
    normal = np.array([-wind[1]/(wind[0]+np.finfo(float).eps), 1])
    normal = normal/np.linalg.norm(normal)
    R = 41

    Z_H = 80  # Hub height of rotor in m
    Z_0 = 0.3  # Hub height of rotor in m
    alpha = 0.5 / (np.log(Z_H / Z_0))


    x_lines, y_lines = [], []
    for i in range(len(x)):
        turbine_pos = np.array([x[i], y[i]])

        pos1 = turbine_pos + np.array([0, R])
        pos2 = turbine_pos - np.array([0, R])

        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]])

        dist = 150*R
        rl = alpha*dist + R

        pos4 = turbine_pos + np.array([dist, rl])
        pos3 = turbine_pos - np.array([0, rl]) + np.array([dist, 0])

        x_pos = [pos1[0], pos2[0], pos3[0], pos4[0]]
        y_pos = [pos1[1], pos2[1], pos3[1], pos4[1]]

        plt.fill(x_pos, y_pos, facecolor='lightsalmon', edgecolor='orangered', alpha=0.25)

def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmin, ymin], [xmax, ymin]])

    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)


def get_turbine_lines_test(x, y, wind):
    normal = np.array([-wind[1]/(wind[0]+np.finfo(float).eps), 1])
    normal = normal/np.linalg.norm(normal)
    R = 41

    Z_H = 80  # Hub height of rotor in m
    Z_0 = 0.3  # Hub height of rotor in m
    alpha = 0.5 / (np.log(Z_H / Z_0))


    x_lines, y_lines = [], []
    for i in range(len(x)):
        turbine_pos = np.array([x[i], y[i]])

        pos1 = turbine_pos + np.array([0, R])
        pos2 = turbine_pos - np.array([0, R])

        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]])

        dist = 10*R
        rl = alpha*dist + R

        pos4 = turbine_pos + np.array([dist, rl])
        pos3 = turbine_pos - np.array([0, rl]) + np.array([dist, 0])

        xx = np.array([pos1[0], pos2[0], pos3[0], pos4[0], pos1[0]])
        yy = np.array([pos1[1], pos2[1], pos3[1], pos4[1], pos1[1]])

        gradient_fill(xx, yy)



        # plt.fill(x_pos, y_pos, facecolor='lightsalmon', edgecolor='orangered', alpha=0.25)




def get_wake_plots(x, y, wind_dir=[12, 0]):
    """
    x, y: 1D arays are turbines coordinates
    wind: 1x2 array of x and y components of the wind.
    """

    get_turbine_lines(x, y, wind_dir)
    plt.scatter(x, y)
    plt.xlim(0, 1200)
    plt.ylim(0, 1200)

    plt.show()
    # plt.plot(x_lines, y_lines)


if __name__ == "__main__":
    x = np.random.random(5) * 1000
    y = np.random.random(5) * 1000

    get_wake_plots(x, y, wind_dir=[0, 12])

