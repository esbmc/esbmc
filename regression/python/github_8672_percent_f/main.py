x = 2.5
if nondet_bool():
    x = 1.5
s = "v %f" % x
assert s.startswith("v ")
