import math


def log_of(x: float, base: float) -> float:
    return math.log(x, base)


assert log_of(16.0, 2.0) > 3.9
log_of(16.0, 1.0)
