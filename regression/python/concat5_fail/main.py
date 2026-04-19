t: str = "foo"
s: str = "bar"
s += t
assert s == "foobar"
assert len(s) == 7

