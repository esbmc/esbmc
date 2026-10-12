def f(k: int):
    return lambda x: x + k


def g():
    n = 4
    return lambda x: x * n


def h(k: int):
    if k > 0:
        return lambda x: x + k
    return lambda x: x - k


y = f(1)
assert y(2) == 3
z = g()
assert z(3) == 12
w = h(2)
assert w(5) == 7
