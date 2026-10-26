# A conditional expression with a name arm does not type a method parameter (#9210).
class C:
    def m(self, v):
        return isinstance(v, str)


c = nondet_bool()
e = nondet_bool()
if c:
    x = 1
else:
    x = "a"
o = C()
assert not o.m(x if e else 0)
