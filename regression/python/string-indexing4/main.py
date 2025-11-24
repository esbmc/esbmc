foo = "abc:xyz:123"

assert foo.startswith("abc:")
assert foo[4:7] == "xyz"
assert foo[8:] == "123"
assert foo[-3:] == "123"

assert foo[:4] == "abc:"
assert foo[4:] == "xyz:123"
assert foo[:3] == "abc"
assert foo[-7:-4] == "xyz"
