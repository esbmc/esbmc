def test_set_difference():
    a = {1, 2, 3, 4}
    b = {3, 4, 5}
    result = a & b  # set difference: elements common to both sets
    assert result == {3}

test_set_difference()
