def test_extend_non_empty():
    a = [1]
    a.extend([2])
    assert a == [1, 2]


test_extend_non_empty()
