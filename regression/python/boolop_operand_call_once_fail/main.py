n = 0


def f() -> int:
    global n
    n = n + 1
    return 5


v = f() or 2
assert v == 5
assert n == 2
