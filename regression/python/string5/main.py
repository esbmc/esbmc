a = str("hello")
b = str("hello")
c = str("world")
assert a == b                  # identical strings
assert a != c                  # different strings
assert "hello" == a            # literal on the left
assert b == "hello"            # literal on the right
assert not (a != "hello")      # negated inequality
assert str("test") == str("test")  # inline str() calls
assert str("one") != str("two")  # unequal literals
