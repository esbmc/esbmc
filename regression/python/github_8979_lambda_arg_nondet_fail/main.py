def f3(a, x):
    return a(x)


e = f3(lambda x: x > 0, nondet_int())
assert isinstance(e, str)
