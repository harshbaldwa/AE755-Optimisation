"""
Based on Optimizing Wind Farm Layout for Maximum Energy
Production Using Direct Methods, Jacob R. West, 2019
"""


from pysph.base import nnps
from pysph.base.utils import get_particle_array
from cyarray.api import UIntArray
import numpy as np


def penalty_function(
    pos, boundary_limits=[[0, 4000], [0, 3500]], diameter=40, rho=1.0
):
    """
    pos: Array containing positions of the turbines. Format - [x1, y1, x2, y2 ...]
    boundary limits: [[x1, x2], [y1, y2]]
    diameter: Diameter of the turbine
    rho: Scaling parameter of the penalty function. (update when more optimal point is obtained)
    """

    x = pos[::2]
    y = pos[1::2]

    if not(len(x) == len(y)):
        print("Number of x coordinates and y cooridnates are different")

    xlimits = boundary_limits[0]
    ylimits = boundary_limits[1]

    beta1 = 0.0

    turbines = get_particle_array(x=x, y=y, h=5 * diameter)
    nps = nnps.LinkedListNNPS(dim=2, particles=[turbines, turbines], radius_scale=1)

    src_index, dst_index = 0, 1
    nps.set_context(src_index=0, dst_index=1)
    nbrs = UIntArray()

    # beta1 calculation
    for i in range(len(x)):
        nps.get_nearest_particles(src_index, dst_index, i, nbrs)
        neighbours = nbrs.get_npy_array()
        for j in range(len(neighbours)):
            distance = np.sqrt(
                (x[i] - x[neighbours[j]]) ** 2 + (y[i] - y[neighbours[j]]) ** 2
            )
            beta1 += distance

    # beta2 calculation
    idx1 = x < xlimits[0]
    idx2 = y < ylimits[0]

    beta2 = abs(x[idx1]).sum() + abs(y[idx2]).sum()

    # beta3 calculation

    idx1 = x > xlimits[1]
    idx2 = y > ylimits[1]

    beta3 = abs(x[idx1] - xlimits[1]).sum() + abs(y[idx2] - ylimits[1]).sum()

    # beta calculation (penalty function)

    beta = (rho * beta1) ** 2 + (rho * (beta2 + beta3) ** 2)

    return beta


# if __name__ == "__main__":
#     # print(check_constraints([0.0, 1.1], [0.0, 0.2]))
#     print(penalty_function([0.0, 0.0, 1.0]))
