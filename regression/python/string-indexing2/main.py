foo = "abc:xyz:123"

assert foo.startswith("abc:")
assert foo[4:7] == "xyz"
assert foo[8:] == "123"
assert foo[-3:] == "123"
