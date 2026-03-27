def test_nondet_list_bool_in_conditional():
    """Test nondet list behavior in conditionals."""
    x: list[bool] = nondet_list(9, nondet_bool())

    result = False

    if len(x) == 0:
        result = False
    elif len(x) == 1:
        result = x[0]
    elif len(x) == 2:
        result = x[0] or x[1]

    assert result == result


test_nondet_list_bool_in_conditional()


def test_nondet_list_in_conditional():
    """Test nondet list behavior in conditionals."""
    x: list[int] = nondet_list(9)

    result = 0

    if len(x) == 0:
        result = 0
    elif len(x) == 1:
        result = x[0]
    elif len(x) == 2:
        result = x[0] + x[1]

    # Result is always defined
    assert result == result


test_nondet_list_in_conditional()
