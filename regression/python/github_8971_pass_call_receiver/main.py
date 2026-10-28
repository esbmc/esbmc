class C:
    def ident(self, x):
        return x


flag = nondet_bool()
v = nondet_int() if flag else nondet_str()
assert C().ident(v) == v
