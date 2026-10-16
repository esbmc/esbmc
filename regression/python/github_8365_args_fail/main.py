def f(a: int) -> int:
    return a


def g() -> int:
    assert False
    return 1


try:
    f(g(), 2)
except TypeError:
    pass
