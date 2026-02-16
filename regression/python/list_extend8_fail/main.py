def test_extend_string_digits():
    a = ['x']
    a.extend("123")
    assert a == ['x', 'y', '2', '3']

test_extend_string_digits()
