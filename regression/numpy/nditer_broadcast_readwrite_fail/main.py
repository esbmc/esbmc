import numpy as np

a = np.array([1])
b = np.broadcast_to(a, (2,))

np.nditer(b, op_flags=["readwrite"])
