def f() -> int:
    return 1


def k() -> int:
    g = a
    return g()


a = f
assert k() == 1
