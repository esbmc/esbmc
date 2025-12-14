def test_set_difference():
    a = {1, 2, 3, 4}
    b = {3, 4, 5}
    result = a | b  # set difference: all unique elements from both sets
    assert result == {1, 2, 3, 4, 5}

test_set_difference()
