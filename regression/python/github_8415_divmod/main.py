count: int = 0


def left() -> int:
    global count
    count += 1
    return -7


def right() -> int:
    global count
    count += 1
    return -5


q, r = divmod(left(), 5)
assert count == 1
assert q == -2
assert r == 3

q, r = divmod(7, right())
assert count == 2
assert q == -2
assert r == -3

q, r = divmod(nondet_int(), 5)
assert r >= 0
assert r < 5
