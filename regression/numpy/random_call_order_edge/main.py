import numpy as np

trace = 0


def low() -> float:
    global trace
    trace = trace * 10 + 1
    return 0.0


def high() -> float:
    global trace
    trace = trace * 10 + 2
    return 1.0


x = np.random.uniform(high=high(), low=low())

assert trace == 21
assert x <= 1.0
