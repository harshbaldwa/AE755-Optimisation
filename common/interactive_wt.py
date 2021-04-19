from .cost import objective
import matplotlib.pyplot as plt
from .wake_visualization import get_wake_plots_interactive
import numpy as np

n = 4 
layout = np.zeros(n*2)
bounds = np.array([[0, 4000], [0, 3500]])
diameter = 82
height=80
z_0 = 0.3
windsp = np.array([12])
theta = np.array([0])
wind_prob = np.array([1])

b_range = 0.2*np.array([bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]])

b1 = bounds[0][0]
b2 = bounds[0][1]
b3 = bounds[1][0]
b4 = bounds[1][1]
plt.plot([b1, b1, b2, b2, b1], [b3, b4, b4, b3, b3], color='white', linestyle="-", linewidth=0.5)

plt.xlim(bounds[0, 0] - b_range[0], bounds[0, 1] + b_range[0])
plt.ylim(bounds[1, 0] - b_range[1], bounds[1, 1] + b_range[1])

for i in range(n):
    layout[2*i:2*(i+1)] = np.array(plt.ginput(2)[0])
    # layout = layout.reshape(2*n)
    print(-objective(layout[:2*(i+1)], bounds, diameter, height, z_0, windsp, theta, wind_prob))
    get_wake_plots_interactive(layout[:2*(i+1):2], layout[1:2*(i+1):2], bounds, diameter, height, z_0, windsp, theta, wind_prob, [])
