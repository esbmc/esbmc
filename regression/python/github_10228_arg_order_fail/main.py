# Issue #10228: 2 ** n is read before a later argument rebinds n
n = 3


def h() -> int:
    global n
    n = 5
    return 0


def g(k: int, z: int) -> int:
    return k


r = g(2 ** n, h())
assert r == 32
