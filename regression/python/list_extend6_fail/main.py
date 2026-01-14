def test_extend_empty_string():
    a = ['x']
    a.extend("")
    assert a == ['y']

test_extend_empty_string()
