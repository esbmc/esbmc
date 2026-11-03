# An int name and a str literal on the arms of a conditional expression.
n = 1
c = nondet_bool()
k = n if c else "a"
assert isinstance(k, str) == (not c)
if c:
    assert k == 1
else:
    assert k == "a"
