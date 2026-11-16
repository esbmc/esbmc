flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
if x:
    assert not (isinstance(x, int) and x == 0)
    assert not (isinstance(x, str) and x == "")
else:
    assert (isinstance(x, int) and x == 0) or (isinstance(x, str) and x == "")
