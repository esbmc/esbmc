import numpy as np

a = np.array(["ab", "c"])
k = np.random.randint(0, 2)
s = a[k]

assert s == "ab"
