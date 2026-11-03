def small():
    return nondet_int() % 10


xs = nondet_list(small)
for v in xs:
    assert v < 9
