class C:
    def ident(self, x):
        return x


flag = nondet_bool()
v = nondet_int() if flag else nondet_str()
c = C()
r = c.ident(v)
assert r != v
