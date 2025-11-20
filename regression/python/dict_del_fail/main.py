my_dict = {"a": 1, "b": 2, "c": 3}
assert (my_dict["a"] == 1)
del my_dict["a"]
assert "b" not in my_dict
