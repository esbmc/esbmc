# dict.update() with mix of new and existing keys
d: dict = {"a": 1, "b": 2, "c": 3}
d.update({"b": 20, "d": 40})
assert d["a"] == 1  # unchanged
assert d["b"] == 20  # updated
assert d["c"] == 3  # unchanged
assert d["d"] == 40  # inserted
