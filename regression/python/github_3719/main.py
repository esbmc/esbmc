def test_het_dict_iteration():
    d = {1: "a", "b": "c"}
    for k in d:
        assert k == 1 or k == "b"

test_het_dict_iteration()
