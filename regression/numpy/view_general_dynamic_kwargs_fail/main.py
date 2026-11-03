import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

def read(**kw):
    w = kw["w"]
    return w[0]

opts = {"w": v}
r = read(**opts)
