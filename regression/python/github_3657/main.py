# Test case from issue #3657: dict.update() basic usage
d = {"a": 1, "b": 2}
d.update({"e": 5, "f": 6})
assert d["e"] == 5
assert d["f"] == 6
