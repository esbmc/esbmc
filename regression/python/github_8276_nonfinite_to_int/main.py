import math


def conv(kind: int, x: float) -> int:
    try:
        if kind == 0:
            return math.floor(x)
        if kind == 1:
            return math.ceil(x)
        if kind == 2:
            return math.trunc(x)
        return round(x)
    except OverflowError:
        return -1
    except ValueError:
        return -2


k: int = 0
while k < 4:
    assert conv(k, float("inf")) == -1
    assert conv(k, float("-inf")) == -1
    assert conv(k, float("nan")) == -2
    k += 1

assert conv(0, 2.5) == 2
assert conv(1, 2.5) == 3
assert conv(2, -2.5) == -2
assert conv(3, 2.5) == 2
