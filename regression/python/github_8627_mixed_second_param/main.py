# Both parameters change type between the calls, so each copy of op types
# its second parameter from the calls that reach it.
def op(a, b):
    return a * b


r1 = op(2, 2.5)
r2 = op(1.5, 2)
assert isinstance(r1, float)
assert isinstance(r2, float)
assert r1 == 5.0
assert r2 == 3.0
