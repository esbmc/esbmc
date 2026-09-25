import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
alias = row
alias[0] = 99

assert a[0][0] == 1
