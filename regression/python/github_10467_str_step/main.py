w = "abcdef"
s = 2
r = w[::s]
assert r == "ace"
t = -2
q = w[::t]
assert q == "fdb"
