import numpy as np

a = np.linspace(0, 255, 4, dtype=np.uint8)
assert a[1] == 85 and a[3] == 255
