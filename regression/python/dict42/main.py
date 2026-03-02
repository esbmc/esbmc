def test_get_with_default():
    d = {"a": 1}
    e = {"b": 2}

    assert e is not d
    assert d.get("a", 0) == 1
    assert d.get("b", 0) == 0
    assert e.get("b", 1) == 2
    assert e.get("a", 2) == 2


test_get_with_default()
