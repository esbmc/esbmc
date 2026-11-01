import numpy as np

def f(s):
    return s[0]


a=np.array(['ab','cd','ef'])
assert f(a[1:])=='cd'
