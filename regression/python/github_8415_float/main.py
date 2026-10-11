count: int = 0


def left() -> float:
    global count
    count += 1
    return -7.5


def right() -> float:
    global count
    count += 1
    return -2.0


r = left() % 2.0
assert count == 1
assert r == 0.5

r = left() // 2.0
assert count == 2
assert r == -4.0

r = 7.5 % right()
assert count == 3
assert r == -0.5

r = 7.5 // right()
assert count == 4
assert r == -4.0
