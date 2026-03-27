def test_nondet_list_str():
    y = nondet_str()
    x = []
    x.append(y)
    assert x[len(x) - 1] == y


test_nondet_list_str()
