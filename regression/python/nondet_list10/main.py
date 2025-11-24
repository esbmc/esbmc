def test_nondet_list_in_conditional():
    """Test nondet list behavior in conditionals."""
    x:list[int] = nondet_list(9)

    result = 0

    if len(x) == 0:
        result = 0
    elif len(x) == 1:
        result = x[0]
    
    # Result is always defined
    assert result == result

test_nondet_list_in_conditional()
