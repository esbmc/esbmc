import numpy as np

def f():
    return 'ab'


a=np.array(['ab','cd'])
assert f() in a
