import random
import math
import numpy as np

class Layout:
	def __init__(self, X, Y, N):
		self.N = N
		layout = np.zeros(2*N)
		layout[::2] = X
		layout[1::2] = Y
		self.layout = layout

def random_layout(N, x_bound, y_bound, diameter):
	grid_size = 5*diameter
	A = int( (x_bound[1] - x_bound[0])/grid_size )
	B = int( (y_bound[1] - y_bound[0])/grid_size )
	random_grids = random.sample(range(A*B), N)
	X = np.zeros(N)
	Y = np.zeros(N)
	for i in range(N):
		row_num = random_grids[i]%A
		col_num = (random_grids[i] - row_num)/A
		X[i] = col_num*grid_size + grid_size/2
		Y[i] = row_num*grid_size + grid_size/2
	
	return Layout(X, Y, N)

