def f1() -> int:
    return 1


def f2() -> int:
    return 2


def f3() -> int:
    return 3


a, *b, c = f1, f2, f3
assert c() == 3
assert a() == 1
