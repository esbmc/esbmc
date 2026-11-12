# As function_value_dict_ints_read, but the "b" path asserts what is false.
d = {"a": 1, "b": 3}
for k in ("a", "b"):
    v = d[k]
    assert v == 1
