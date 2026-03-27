def test_list_assignments():
    x = [1, 2, 3]
    y = [1, 2, 3]
    assert x is not y  # Different objects with same content
    z = y
    assert y is z      # Same object reference
    assert x is not z  # Different objects

test_list_assignments()
