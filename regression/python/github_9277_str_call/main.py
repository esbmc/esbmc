# An int expression and str() of it on the arms of a conditional expression.
n = abs(nondet_int()) % 3
c = nondet_bool()
k = n if c else str(n)
if c:
    assert isinstance(k, int)
else:
    assert isinstance(k, str)
assert isinstance(k, str) == (not c)
