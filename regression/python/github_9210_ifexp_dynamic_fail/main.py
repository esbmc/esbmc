# A conditional expression over an int-or-str name does not type its parameter (#9210).
def f(v):
    return isinstance(v, str)


c = nondet_bool()
d = nondet_bool()
if c:
    x = 1
else:
    x = "a"
r = f(x if d else 0)
assert not r
