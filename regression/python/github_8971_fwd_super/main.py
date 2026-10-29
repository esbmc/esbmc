class Base:
    def __init__(self, a):
        self.a = a


class Derived(Base):
    def __init__(self, a):
        super().__init__(a)

    def read(self):
        return self.a


flag = nondet_bool()
x = nondet_str() if flag else nondet_int()
d = Derived(x)
r = d.read()
assert r is x
assert isinstance(r, str) == flag
