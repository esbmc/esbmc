# A str compared with a conditional choosing between two str literals.
s = nondet_str()
c = nondet_bool()
if s == ("a" if c else "b"):
    assert len(s) == 1
