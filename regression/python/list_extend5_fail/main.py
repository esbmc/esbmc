def test_extend_string():
    a = ['a']
    a.extend("bc")
    assert a == ['a', 'b', 'b']

test_extend_string()
