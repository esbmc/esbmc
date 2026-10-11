count: int = 0


def f() -> int:
    global count
    count += 1
    return -7


i = 0
while f() % 5 != 3 and i < 3:
    i += 1
assert count == 1
assert i == 0

b = False
r = b and f() ** 2 == 49
assert count == 1
r = not b and f() ** 2 == 49
assert r
assert count == 2
