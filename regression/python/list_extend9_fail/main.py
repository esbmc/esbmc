def test_extend_string_with_space():
    a = ['a']
    a.extend("b")
    assert a == ['a', ' ', 'b']


test_extend_string_with_space()
