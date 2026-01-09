def test_dict_keys():
    d = {"a": 1, "b": 2}

    assert set(d.keys()) == {"a", "c"}

test_dict_keys()
