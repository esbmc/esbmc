def f():
    return 1


d = nondet_dict(2)
d[0] = f
assert d[0]() == 1
