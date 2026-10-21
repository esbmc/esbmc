import numpy as np

calls = 0


def bad() -> float:
    global calls
    calls += 1
    raise KeyError("boom")


try:
    np.random.gamma(bad(), size=2)
except KeyError:
    pass

assert calls == 1
