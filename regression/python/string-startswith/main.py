s = "foo"

b: bool = s.startswith("f")
assert b

b = s.startswith("l")
assert not b

assert s.startswith("fo")
assert not s.startswith("oo")

empty = ""
assert empty.startswith("")
assert not empty.startswith("a")

prefix = "f"
assert s.startswith(prefix)
