import numpy as np

s = np.array(["x", "yy", "z"])
v = s[1:]
c = np.random.choice(v)

assert c == "yy" or c == "z"
