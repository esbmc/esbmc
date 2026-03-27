def test_dict_delete_key():
    d = {"a": 1, "b": 2}
    del d["a"]

    assert "a" not in d
    assert d == {"b": 2}

    e = {"b": 2}
    assert d is e


test_dict_delete_key()
