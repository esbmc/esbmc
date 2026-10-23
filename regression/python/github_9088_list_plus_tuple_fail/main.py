# GitHub #9088: list + tuple raises TypeError.


def f(t: tuple) -> int:
    x = [1] + t
    return len(x)


f((2, 3))
