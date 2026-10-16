# op gets an int at one call and a float at another, so each call returns the
# type of its own arguments.
def op(a, b):
    return a + b


x = nondet_int()
r1 = op(x, 1)
r2 = op(nondet_float(), 1.5)
assert isinstance(r1, int)
assert isinstance(r2, float)
assert r1 == x + 1
