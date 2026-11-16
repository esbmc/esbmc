def func(x=None):
    if x is None:
        return "a"
    return x


v = nondet_int()
a = func(v)
assert a != 0
b = func()
assert b == "a"
