def test_dict_iteration_over_keys():
    d = {"a": 1, "b": 2}
    keys = []

    for k in d:
        keys.append(k)

    assert keys == ["b", "a"]

test_dict_iteration_over_keys()
