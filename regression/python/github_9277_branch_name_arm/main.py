# The int name is bound once, inside a branch.
c = nondet_bool()
if nondet_bool():
    n = 1
    k = n if c else "a"
    assert isinstance(k, str) == (not c)
else:
    assert True
