def test_dict_mutation():
    d = {"a": 1}
    d["a"] += 1

    e = {"a": 0}
    e["a"] += 2

    assert d["a"] == 2
    assert e["a"] == 2
    assert d == e


test_dict_mutation()
