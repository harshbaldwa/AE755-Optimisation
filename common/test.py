import wake_model
import numpy as np
import layout as lyt
import matplotlib.pyplot as plt

#layout_1 = lyt.random_layout(1, [0, 2000], [0, 2000], 200).layout
#layout_1 = lyt.Layout(np.ones(30)*1000, np.linspace(100, 1900, 30), 30).layout
layout_1 = lyt.Layout( np.linspace(100, 1900, 30),np.ones(30)*1000, 30).layout
alpha = 0.5/(np.log(60/0.3))


AEP = wake_model.aep(layout_1, [12, 0], alpha, 20)
print(AEP/(365*24*60*60*1e6))
plt.plot(layout_1[::2], layout_1[1::2], 'ro')
plt.show()
