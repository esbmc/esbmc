# A function body passes a function defined after it as a value (#8369).
def apply(fn, x: int) -> int:
    return fn(x)


def f(x: int) -> int:
    return apply(g, x)


def g(x: int) -> int:
    return x + 1


assert f(1) == 2
