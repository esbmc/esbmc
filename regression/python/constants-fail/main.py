X = 2**4
if nondet_bool():
    assert X == 15

SIZE = uint64(10)
if nondet_bool():
    assert SIZE == 9
