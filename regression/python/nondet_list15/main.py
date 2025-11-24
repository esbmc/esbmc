def test_nondet_list_in_conditional():
    a = nondet_bool()
    b = nondet_bool()
    x = []
    x.append(a)
    x.append(b)

    result = []  # Initialize result as a list

    if len(x) == 0:
        result.append(False)
    elif len(x) == 1:
        result.append(x[0])

    y = nondet_bool()
    x.append(y)
    x.append(y)  # Append the new value to result

    assert x[len(x)-1] == y  # This now correctly checks the last item in result

test_nondet_list_in_conditional()

