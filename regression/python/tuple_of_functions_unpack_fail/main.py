def f1() -> int:
    return 1


def f2() -> int:
    return 2


fs = (f1, f2)
a, b = fs
assert b() == 1
