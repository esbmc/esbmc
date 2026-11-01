# Issue #9446: s[1] is "b", so the assertion fails.
s: str = "abcd"
r: str = "?"
if len(s) > 2:
    r = s[1]
else:
    r = "?"
assert r == "c"
