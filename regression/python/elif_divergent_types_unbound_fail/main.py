c = nondet_bool()
if c:
    a = "Hello"
if c:
    assert a == "Hello"
else:
    assert isinstance(a, int)
