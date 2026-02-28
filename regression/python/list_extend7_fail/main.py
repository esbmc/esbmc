def test_extend_repeated_chars():
    a = []
    a.extend("aaa")
    assert a == ['a', 'a']


test_extend_repeated_chars()
