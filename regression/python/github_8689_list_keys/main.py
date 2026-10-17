# list() of a dict or of its keys view is a plain list in insertion order (#8689).
d = {"a": 1, "b": 2}
assert list(d.keys()) == ["a", "b"]
assert list(d) == ["a", "b"]
assert list(d.keys()) != ["b", "a"]

e = {1: "x", 2: "y"}
assert list(e) == [1, 2]
f = {2: "y", 1: "x"}
assert list(f) != list(e)
assert list(f.keys()) != list(e.keys())
