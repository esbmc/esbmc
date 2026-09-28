cond = nondet_bool()
if cond:
    x = "a"
    assert x < "b"
    x = "ab"
    assert x < "abc"
    x = "abc"
    assert not (x < "ab")
else:
    x = 1
