def f1() -> int:
    return 1


def f2() -> int:
    return 2


a, b = c, d = f1, f2
assert c() == 2
