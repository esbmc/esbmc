def f(x):
    return x * 2


a = f(abs(-1.5) + 1)
b = f(abs(-1.5))
assert b == 2
