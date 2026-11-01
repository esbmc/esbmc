import numpy as np

a = np.array(["ab"])
a[0] = "\u00e91"
assert a[0] != "\u00e91"
