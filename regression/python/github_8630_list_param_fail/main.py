def f(a):
    return a + 1


def h(xs: list) -> int:
    return f(xs[0])


assert h([1, 2]) == 3
