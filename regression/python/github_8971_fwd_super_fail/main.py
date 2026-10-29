class Base:
    def __init__(self, a):
        self.a = a


class Derived(Base):
    def __init__(self, a):
        super().__init__(a)


flag = nondet_bool()
x = nondet_str() if flag else nondet_int()
d = Derived(x)
assert isinstance(d.a, str) == (not flag)
