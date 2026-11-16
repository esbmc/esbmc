flag = nondet_bool()
zero = 0.0 if flag else ""
one = 1.5 if flag else "a"
if zero:
    assert False
assert not zero
assert one
if not one:
    assert False
