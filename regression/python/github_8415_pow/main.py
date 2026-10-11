count: int = 0


def base() -> int:
    global count
    count += 1
    return 3


r = base() ** 3
assert count == 1
assert r == 27

r = base() ** 0
assert count == 2
assert r == 1

r = pow(base(), 3)
assert count == 3
assert r == 27
