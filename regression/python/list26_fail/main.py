def test_nondet_list_in_conditional():
    a = False
    b = False
    x = []
    x.append(a)
    x.append(b)

    result = []

    if len(x) == 0:
        result.append(False)
    elif len(x) == 1:
        result.append(x[0])

    y = False
    x.append(y)
    result.append(y)

    assert result[len(x) - 1] == y


test_nondet_list_in_conditional()
