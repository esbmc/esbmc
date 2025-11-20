foo = "abc:xyz:123"
assert foo[0:0] == "a"        # Should be empty
assert foo[::2] == "abc"      # Wrong step slicing
