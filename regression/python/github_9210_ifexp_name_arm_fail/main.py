# A conditional expression with a name arm does not type its parameter (#9210).
def f(v):
    return isinstance(v, str)


c = nondet_bool()
e = nondet_bool()
x = 1
if c:
    x = "abc".upper()
r = f(x if e else 0)
assert not r
