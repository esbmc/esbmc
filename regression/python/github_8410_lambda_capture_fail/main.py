def f(k: int):
    return lambda x: x + k


y = f(1)
assert y(2) == 4
