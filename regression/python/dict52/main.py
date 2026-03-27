def test_dict_iteration_over_keys():
    d = {"a": 1, "b": 2}
    keys = []
    l: list[str] = d.keys()
    for k in l:
        keys.append(k)

    assert set(keys) == {"a", "b"}


test_dict_iteration_over_keys()
