x = 2
if nondet_bool():
    x = 1
assert str(x) != "2"
