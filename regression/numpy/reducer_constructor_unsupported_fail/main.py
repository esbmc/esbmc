import numpy as np

# std()/var() do not support axis/ddof/keepdims/where/out/dtype kwargs yet;
# this must keep the existing explicit diagnostic for a constructor array
# too, not silently ignore the keyword or misread it as data.
a = np.identity(2)
b = np.std(a, axis=0)

assert b[0] == 0
