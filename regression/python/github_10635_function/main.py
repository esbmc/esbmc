def f() -> int:
    return 1


def k() -> int:
    return a()


a = f
assert k() == 1
