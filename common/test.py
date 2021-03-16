from .wake_model import aep
import numpy as np
from .layout import random_layout as lyt
from .cost import obj
import matplotlib.pyplot as plt

# Random layout
layout_1 = lyt(15, [0, 4000], [0, 3500], 82*5).layout
#All the turbines in one vertical
# layout_1 = lyt(np.ones(100)*1000, np.linspace(100, 1900, 100), 100).layout
# All in one horizontal
# layout_1 = lyt( np.linspace(100, 1900, 30),np.ones(30)*1000, 30).layout
alpha = 0.5/(np.log(80/0.3))


AEP = aep(layout_1, [12, 0], alpha, 41)
print(AEP/(365*24*60*60*1e6))
print(obj(layout_1, [0, 4000], [0, 3500]))
# plt.plot(layout_1[::2], layout_1[1::2], 'ro')
# plt.show()
