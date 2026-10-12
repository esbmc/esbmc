# Calling the function before the module defines the function it passes
# raises NameError in CPython (#8369).
def apply(fn, x: int) -> int:
    return fn(x)


def f(x: int) -> int:
    return apply(g, x)


assert f(1) == 2


def g(x: int) -> int:
    return x + 1
