def test_empty_difference():
    a = set()
    b = set()
    result = a - b
    assert result == set()


test_empty_difference()
