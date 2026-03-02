def test_nondet_list_in_conditional():
    a = False
    b = False
    x = []
    x.append(a)
    x.append(b)
    y = False
    x.append(y)
    assert x[len(x) - 1] == y


test_nondet_list_in_conditional()
