from .wake_model import aep
import numpy as np
from .layout import Layout as lyt
from .cost import obj
import matplotlib.pyplot as plt

# Random layout
# layout_1 = lyt(15, [0, 4000], [0, 3500], 82*5).layout
#All the turbines in one vertical
# layout_1 = lyt(np.ones(4)*1000, np.linspace(100, 1900, 4), 4).layout
# All in one horizontal
# layout_1 = lyt( np.linspace(100, 1900, 30),np.ones(30)*1000, 30).layout
# alpha = 0.5/(np.log(80/0.3))


# AEP = aep(layout_1, [12, 0], alpha, 41, [[0, 4000], [0, 3500]])
# AEP, _ = aep(layout_1, [12, 0], alpha, 41)
# print(AEP)
# print(obj(layout_1, [0, 2000], [0, 2000]))
# plt.plot(layout_1[::2], layout_1[1::2], 'ro')
# plt.show()