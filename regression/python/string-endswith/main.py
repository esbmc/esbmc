s = "foo"

b: bool = s.startswith("f")
assert b
b = s.startswith("l")
assert not b
assert s.startswith("fo")
assert not s.startswith("oo")

b = s.endswith("o")
assert b
b = s.endswith("x")
assert not b
assert s.endswith("oo")
assert not s.endswith("fo")

empty = ""
assert empty.startswith("")
assert not empty.startswith("a")
assert empty.endswith("")
assert not empty.endswith("a")

prefix = "f"
assert s.startswith(prefix)
suffix = "o"
assert s.endswith(suffix)

assert s.startswith("foo")
assert s.endswith("foo")

single = "x"
assert single.startswith("x")
assert single.endswith("x")
assert not single.startswith("y")
assert not single.endswith("y")

