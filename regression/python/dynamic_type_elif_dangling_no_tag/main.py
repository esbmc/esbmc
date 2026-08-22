# No final else, so x should stay a plain int, not get tagged.
cond1 = nondet_bool()
cond2 = nondet_bool()
if cond1:
    x = 1
    assert x == 1
elif cond2:
    x = 2
    assert x == 2
