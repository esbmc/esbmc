def test_het_dict_values():
    d = {"a": 1, "b": 2.0}
    for v in d.values():
        assert v == 1 or v == 2.0

test_het_dict_values()
