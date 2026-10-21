import numpy as np

count = 0


def one() -> int:
    global count
    count += 1
    return 1


a = np.random.choice([one(), 2], size=3)

assert count == 1
assert a[2] == 1 or a[2] == 2
