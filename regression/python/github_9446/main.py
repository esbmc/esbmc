# Issue #9446: a str index and a literal joined into one variable.
s: str = "abcd"
r: str = "?"
if len(s) > 2:
    r = s[1]
else:
    r = "?"
assert r == "b"

q: str = "?"
j = 2
try:
    q = s[j]
except IndexError:
    q = "?"
assert q == "c"

w: str = "xyzw"
if len(s) > 2:
    w = s[-1]
else:
    w = "?"
assert w == "d" and len(w) == 1
