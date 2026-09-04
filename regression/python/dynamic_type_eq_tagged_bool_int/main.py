# bool and int compare equal to each other in real Python.
cond = nondet_bool()
if cond:
    x = True
    y = 1
else:
    x = "a"
    y = "b"
if cond:
    assert x == y
