# Issue #9495: a list read from a tuple by index keeps mixed element types.
t = ("x", [1.0, "s"])
p = t[1]
a, b = p
assert a == 1.0
assert b == "s"
