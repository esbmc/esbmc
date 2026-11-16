class C:
    def n(self):
        return 5

    def f(self, a):
        if a == 0:
            return "s"
        else:
            return self.n()

c = C()
u = nondet_bool()
d = c.f(0 if u else 1)
if not u:
    assert isinstance(d, int)
    assert d == 5
else:
    assert isinstance(d, str)
