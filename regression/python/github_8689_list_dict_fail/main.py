# list() of a dict equals the list of its keys, so != fails (#8689).
d = {"a": 1, "b": 2}
assert list(d) != ["a", "b"]
