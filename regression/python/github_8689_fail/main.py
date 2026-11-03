d = {"a": 1, "b": 2}
e = {1: "x", 2: "y"}
assert list(e) == [1, 2]
assert list(d.keys()) != ["a", "b"]
