def f(k: int):
    g = lambda x: x + k
    k = 10
    return g


y = f(1)
assert y(2) == 3
