a = str("hello")
b = str("world")
c = str("hello")

assert a != c                  # should fail: a == c
assert b == "not world"        # should fail: b != "not world"
assert str("foo") != "foo"     # should fail: equal strings
assert "" != str("")           # should fail: both empty
