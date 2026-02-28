def test_set_difference():
    a = {1, 2, 3, 4}
    b = {3, 4, 5}
    result = a - b  # set difference: elements in a but not in b
    assert result == {1, 2}


test_set_difference()
