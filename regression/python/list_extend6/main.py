def test_extend_empty_string():
    a = ['x']
    a.extend("")
    assert a == ['x']

test_extend_empty_string()
