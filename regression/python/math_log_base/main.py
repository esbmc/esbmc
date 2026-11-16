import math


def log_of(x: float, base: float) -> float:
    return math.log(x, base)


assert abs(math.log(8.0, 2.0) - 3.0) < 1e-9
assert math.log(100, 10) == 2.0
assert math.log(1, 10) == 0.0
assert log_of(16.0, 2.0) > 3.9 and log_of(16.0, 2.0) < 4.1

try:
    math.log(2.0, 1.0)
    assert False
except ZeroDivisionError:
    pass

try:
    math.log(2.0, -2.0)
    assert False
except ValueError:
    pass

try:
    log_of(0.0, 2.0)
    assert False
except ValueError:
    pass
