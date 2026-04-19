foo = "abc:xyz:123"
assert foo.startswith("abc:")
# Ensure "xyz" is present after the colon
assert foo[4:7] == "xyz"
