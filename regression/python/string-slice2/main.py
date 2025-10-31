s: str = "foo"
l = len(s)
ss: str = s[1:l]

assert ss == "oo"
l2 = len(ss)
assert l2 == 2
