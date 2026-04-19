foo = "abc:xyz:123"

assert foo.startswith("abc:")
assert foo[4:7] == "xyzz"
assert foo[8:] == "1232"
assert foo[-3:] == "1232"
