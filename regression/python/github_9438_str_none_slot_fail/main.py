# github issue 9438: a None-typed list slot later given 0 or 0.0 is not None
l = [None]
l[0] = 0
assert str(l[0]) == "None"
m = [None]
m[0] = 0.0
assert repr(m[0]) == "None"
