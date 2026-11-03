def f(x):
    if x == 1:
        return x
    return x * f(x - 1)


a = f(abs(nondet_int()) % 3 + 1)
b = f(2)
assert a == b
