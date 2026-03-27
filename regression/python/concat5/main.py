t: str = "foo"
s: str = "bar"
s += t
assert s == "barfoo"
assert len(s) == 6

