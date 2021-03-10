from pysph.base import nnps
from pysph.base.utils import get_particle_array
from cyarray.api import UIntArray
import numpy as np

def check_constraints(x, y, boundary_limits=[[-10.0, 10.0], [-10.0, 10.0]], diameter=0.2):

    x = np.array(x)
    y = np.array(y)

    xlimits = boundary_limits[0]
    ylimits = boundary_limits[1]

    if (x.min() < xlimits[0]) or (x.max() > xlimits[1]):
        return False

    if (y.min() < ylimits[0]) or (y.max() > ylimits[1]):
        return False

    turbines = get_particle_array(x=x, y=y, h=5*diameter)
    nps = nnps.LinkedListNNPS(dim=2, particles=[turbines, turbines], radius_scale=1)

    src_index, dst_index = 0, 1
    nps.set_context(src_index=0, dst_index=1)
    nbrs = UIntArray()

    for i in range(len(x)):
        nps.get_nearest_particles(src_index, dst_index, i, nbrs)
        neighbours = nbrs.get_npy_array()
        for j in range(len(neighbours)):

            if turbines.x[neighbours[j]] == turbines.x[i] and \
                 turbines.y[neighbours[j]] == turbines.y[i]:
                    pass

            else:
                return False

    return True

if __name__ == '__main__':
    # print(check_constraints([0.0, 1.1], [0.0, 0.2]))
    print(check_constraints([0.0, 0.0, 1.0], [5.1, -5.7, 8.0]))
