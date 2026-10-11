count: int = 0


def left() -> float:
    global count
    count += 1
    return -7.5


r = left() % 2.0
assert r == 0.5
assert count == 2
