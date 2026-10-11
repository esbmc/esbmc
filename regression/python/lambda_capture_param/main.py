def f(k: int):
    return lambda x: x + k


def g(k: int):
    j = k * 2
    return lambda x: x - j


y = f(1)
assert y(2) == 3
h = g(3)
assert h(10) == 4
