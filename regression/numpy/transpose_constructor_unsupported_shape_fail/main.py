import numpy as np

# transpose is explicitly limited to up to 2D arrays; a 3D constructor array
# must hit that specific diagnostic (not a crash, and not the generic
# "Unsupported Numpy call" fallback that non-literal constructors used to
# fall through to before shape materialization was fixed).
a = np.zeros((2, 2, 2))
b = np.transpose(a)

assert b[0][0][0] == 0
