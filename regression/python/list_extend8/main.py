def test_extend_string_digits():
    a = ['x']
    a.extend("1")
    assert a == ['x', '1']


test_extend_string_digits()
