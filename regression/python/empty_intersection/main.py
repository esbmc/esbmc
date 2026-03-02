def test_empty_intersection():
    a = set()
    b = set()
    result = a & b
    assert result == set()


test_empty_intersection()
