# Test: Escape sequences - deve PASSAR
newline = "hello\nworld"
assert len(newline) == 11  # h,e,l,l,o,\n,w,o,r,l,d
tab = "a\tb"
assert len(tab) == 3  # a,\t,b
