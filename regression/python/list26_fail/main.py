def test_nondet_list_in_conditional():
    a = False
    b = False
    x = []
    x.append(a)
    x.append(b)

    result = []  # Initialize result as a list

    if len(x) == 0:
        result.append(False)
    elif len(x) == 1:
        result.append(x[0])

    y = False
    x.append(y)
    result.append(y)  # Append the new value to result

    assert result[len(x)-1] == y  # This now correctly checks the last item in result

test_nondet_list_in_conditional()

