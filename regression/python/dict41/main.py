def test_missing_key_raises_keyerror():
    d = {"a": 1}
    e = {"b": 2}

    assert d != e

    try:
        _ = d["b"]
        assert False  # should not reach here
    except KeyError:
        assert True


test_missing_key_raises_keyerror()
