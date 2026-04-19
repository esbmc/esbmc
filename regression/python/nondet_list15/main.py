def test_nondet_list_in_conditional():
    a = nondet_bool()
    b = nondet_bool()
    x = []
    x.append(a)
    x.append(b)

    result = []

    if len(x) == 0:
        result.append(False)
    elif len(x) == 1:
        result.append(x[0])

    y = nondet_bool()
    x.append(y)
    x.append(y)

    assert x[len(x)-1] == y

test_nondet_list_in_conditional()

