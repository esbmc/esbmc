c = 0


def f() -> int:
    global c
    c += 1
    return -7


def g() -> int:
    global c
    c = c * 10
    return 5


r = f() % 5
assert c == 1
assert r == 3

q = f() // 5
assert c == 2
assert q == -2

c = 1
r = f() % g()
assert c == 20
assert r == 3
