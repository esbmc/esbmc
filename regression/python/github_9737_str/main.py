c = True
q = ["a"] if c else ["b", "c"]
assert isinstance(q[0], str)
assert q == ["a"]
