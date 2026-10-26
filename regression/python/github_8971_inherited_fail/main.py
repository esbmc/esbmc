class Base:
    def __init__(self, v):
        self.v = v


class Derived(Base):
    def get(self):
        return self.v


use_int = nondet_bool()
if use_int:
    x = nondet_int()
else:
    x = "text"
d = Derived(x)
r = d.get()
assert r != x
