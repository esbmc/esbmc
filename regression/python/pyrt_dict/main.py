d = {"a": 1, "b": 2}
assert len(d) == 2
assert d["a"] == 1
d["c"] = 3
d["a"] = 9
assert d["a"] == 9
assert len(d) == 3
assert "c" in d
assert "z" not in d
assert isinstance(d, dict)
assert type(d) is dict
assert not {}
assert d
