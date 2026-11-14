def inc(v):
    return v + 1


x = nondet_int()
if nondet_bool():
    a = x
else:
    a = float(x % 1000)
r = inc(a)
assert r - 1 == a
