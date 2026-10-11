n = nondet_int()
def g():
    return n
def func(a):
    return a()
b = func(g)
assert isinstance(b, int)
