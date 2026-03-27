def test_nondet_list_str():
    x = []
    x.append(nondet_str())
    x.append(nondet_str())
    y = nondet_str()
    x.append(y)
    assert x[len(x) - 1] == "foo"


test_nondet_list_str()
