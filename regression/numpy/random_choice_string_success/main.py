import numpy as np

s = np.random.choice(["ab", "c"])

assert s == "ab" or s == "c"
