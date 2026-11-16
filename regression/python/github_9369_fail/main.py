# Issue #9369: str() of a list was the digits of a pointer, so this was proved.
# CPython gives "[1, 2]", and the assertion fails.
r = (str([1, 2]) == "[1, 2]")
assert not r
