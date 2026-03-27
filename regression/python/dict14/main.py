def test_dict_assignment():
    """Should PASS: Assigning dict to variable"""
    original: dict = {'a': 1, 'b': 2}
    copy = original

    assert copy['a'] == original['a']
    assert copy['b'] == original['b']


test_dict_assignment()
