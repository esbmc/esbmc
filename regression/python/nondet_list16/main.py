def test_nondet_list_in_conditional():
    x = nondet_list(8, nondet_bool())
    y = nondet_bool()
    x.append(y)
    x.append(y)
    assert x[len(x) - 1] == y


test_nondet_list_in_conditional()
