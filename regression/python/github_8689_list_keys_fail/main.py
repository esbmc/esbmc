# list() of a keys view equals the list of keys, so != fails (#8689).
d = {"a": 1, "b": 2}
assert list(d.keys()) != ["a", "b"]
