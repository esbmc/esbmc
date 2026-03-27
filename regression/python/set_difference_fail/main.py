def test_set_difference():
    a = {1, 2}
    b = {3}
    result = a - b  # set difference: elements in a but not in b
    assert result == {1}


test_set_difference()
