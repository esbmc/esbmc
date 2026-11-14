x = lambda x: x + x // x
if nondet_bool():
    v = nondet_int()
    a = x(v)
else:
    v = float(nondet_int() % 1000)
    a = x(v)
assert a - 1 == v
