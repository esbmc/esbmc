count: int = 0
order: int = 0


def left() -> int:
    global count, order
    count += 1
    order = order * 10 + 1
    return -7


def right() -> int:
    global count, order
    count += 1
    order = order * 10 + 2
    return -5


r = left() % 5
assert count == 1
assert r == 3

r = left() // 5
assert count == 2
assert r == -2

r = 7 % right()
assert count == 3
assert r == -3

r = 7 // right()
assert count == 4
assert r == -2

order = 0
r = left() % right()
assert count == 6
assert order == 12
assert r == -2

order = 0
r = left() / right()
assert count == 8
assert order == 12
assert r == 1.4

x = 7
x %= right()
assert count == 9
assert x == -3

x = -7
x //= right()
assert count == 10
assert x == 1

b = False
t = b and left() % 5 == 3
assert count == 10
t = not b and left() % 5 == 3
assert t
assert count == 11
