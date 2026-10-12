c = 0


def f() -> int:
    global c
    c += 1
    return -7


r = f() % 5
q = f() // 5
assert c == 2
assert r == 3
assert q == -2
assert c == 3
