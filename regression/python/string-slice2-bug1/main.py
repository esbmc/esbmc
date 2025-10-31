s: str = "foo"
l = len(s)
ss: str = s[1:l]

assert ss == "oop"
l2 = len(ss)
assert l2 == 2
