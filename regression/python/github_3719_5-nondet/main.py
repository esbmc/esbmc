x = nondet_int()
y = nondet_float()


def test_het_dict_values():
    d = {"a": x, "b": y}
    for v in d.values():
        assert v == x or v == y


test_het_dict_values()
