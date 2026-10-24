def conv(x: float) -> int:
    try:
        return int(x)
    except OverflowError:
        return -1
    except ValueError:
        return -2


assert conv(float("inf")) == -1
assert conv(float("-inf")) == -1
assert conv(float("nan")) == -2
assert conv(3.9) == 3

calls: int = 0


def g() -> float:
    global calls
    calls += 1
    return 2.5


assert int(g()) == 2
assert calls == 1
assert int(g() + 0.5) == 3
assert calls == 2
