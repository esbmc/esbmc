def test_dict_length():
    d = {"a": 1, "b": 2, "c": 3}
    e = {"c": 3, "b": 2, "a": 1}
    assert len(d) == len(e)
    assert d == e

test_dict_length()
