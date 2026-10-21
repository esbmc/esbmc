import numpy as np

count = 0


def scale() -> float:
    global count
    count += 1
    return -1.0


try:
    np.random.normal(0.0, scale(), size=3)
except ValueError:
    pass

assert count == 1
