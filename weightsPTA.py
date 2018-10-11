import numpy as np

"""Initialises weights for the Perceptron and saves it in a .npy file"""

w_0 = np.random.uniform(-1, 1, 1)
w_1 = np.random.uniform(-1, 1, 1)
w_2 = np.random.uniform(-1, 1, 1)
w_array = np.array([w_0, w_1, w_2])
np.save('ptaWeights', w_array)
print w_array
