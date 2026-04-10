def test_dict_iteration_items():
    d = {"a": 1, "b": 2}
    sum_vals = 0

    for v in d.values():
        sum_vals += v

    assert sum_vals == 3

test_dict_iteration_items()
