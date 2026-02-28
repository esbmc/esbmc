def test_dict_contains_key():
    d = {"x": 10, "y": 20}

    assert "x" in d
    assert "z" not in d


test_dict_contains_key()
