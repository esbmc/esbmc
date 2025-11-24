def test_nondet_list_in_conditional():
    x = nondet_list(8, nondet_bool())
    y = nondet_bool()
    x.append(y)
    x.append(y)  # Append the new value to result
    assert x[len(x)-1] == y  # This now correctly checks the last item in result

test_nondet_list_in_conditional()

