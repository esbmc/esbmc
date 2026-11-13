x = nondet_int()
if nondet_bool():
    a = x
else:
    a = 0.5
a = a + 1
assert a > 1 or a <= 1
