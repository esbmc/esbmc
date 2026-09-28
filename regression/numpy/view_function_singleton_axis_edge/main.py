import numpy as np

a = np.array([[[5]]])
v = np.squeeze(a)

assert v == 5
