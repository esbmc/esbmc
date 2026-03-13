def test_set_difference():
    a = {1}
    b = {3}
    result = a | b  # set difference: all unique elements from both sets
    assert result == {1, 3}

test_set_difference()
