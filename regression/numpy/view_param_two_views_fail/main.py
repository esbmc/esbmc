import numpy as np

def copy_first(src, dst):
    dst[0] = src[0]
    dst[1] = src[1]

a = np.array([10, 20, 30])
b = np.array([0, 0, 0])
va = a[0:]
vb = b[0:]
copy_first(va, vb)
assert b[0] == 0  # wrong: copy_first wrote src[0]=10 into dst[0]
