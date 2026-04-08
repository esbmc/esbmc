def test_set_union():
    a = {1}
    b = {3}
    result = a | b  # set union: all unique elements from both sets
    assert result == {1, 2}

test_set_union()
