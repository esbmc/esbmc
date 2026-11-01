import numpy as np

a=np.array(['ab'])


def f(s: str):
    a[0]=s


f('c')
