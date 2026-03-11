# dict.update() should overwrite existing values; this assert must fail
d = {"a": 1, "b": 2}
d.update({"a": 99})
assert d["a"] == 1  # wrong: update should have changed it to 99
