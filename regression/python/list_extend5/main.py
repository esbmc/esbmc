def test_extend_string():
    a = ['a']
    a.extend("bc")
    assert a == ['a', 'b', 'c']


test_extend_string()
