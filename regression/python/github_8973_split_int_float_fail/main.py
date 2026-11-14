x = nondet_int()
if nondet_bool():
    a = x
else:
    a = float(x % 1000)
r = a + 1
assert isinstance(r, int)
