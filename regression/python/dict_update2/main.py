# dict.update() overwrites values for existing keys
d: dict = {"a": 1, "b": 2}
d.update({"a": 100, "b": 200})
assert d["a"] == 100
assert d["b"] == 200
