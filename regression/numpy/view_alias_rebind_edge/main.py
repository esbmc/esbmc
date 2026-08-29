import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
alias = row
row = np.array([0, 0])
alias = np.array([7, 8])
a[0][0] = 99

assert True
