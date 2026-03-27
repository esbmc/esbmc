def test_extend_with_empty():
    a = [1, 2, 3]
    a.extend([])
    assert a == [1, 2, 3]

test_extend_with_empty()
