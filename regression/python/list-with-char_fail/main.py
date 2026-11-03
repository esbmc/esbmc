x = ['1', '2', '3']
if nondet_bool():
    assert len(x) == 2
if nondet_bool():
    assert x[0] == '2'
