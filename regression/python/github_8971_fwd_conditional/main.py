class Base:
    def __init__(self, a):
        self.a = a


class Derived(Base):
    def __init__(self, a):
        if a:
            super().__init__(a)

    def read(self):
        return self.a


flag = nondet_bool()
x = "" if flag else 0
d = Derived(x)
d.read()
