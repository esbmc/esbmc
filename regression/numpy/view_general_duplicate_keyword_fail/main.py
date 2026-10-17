import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

def read(w, k):
    t = w[k]
    return t

r = read(v, w=v, k=0)
