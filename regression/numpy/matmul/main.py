import numpy as np

x = np.dot(3, 4)
assert x == 12

a = np.array([[1,0],[0,1]])
b = np.array([[4,1],[2,2]])
c = np.matmul(a,b)

assert c[0][0] == 4
assert c[0][1] == 1
assert c[1][0] == 2
assert c[1][1] == 2
