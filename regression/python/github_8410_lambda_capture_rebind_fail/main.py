def f(k: int):
    return lambda x: x + k


a = f(1)
b = f(7)
assert a(2) == 9
