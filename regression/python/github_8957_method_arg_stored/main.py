# A bound method is also stored and called; the lift leaves that use as it is.
class K:
    def f(self):
        return "s"

    def g(self):
        return 1

    def call(self, x):
        return x()


a = K()
c = nondet_bool()
h = a.f
r = a.call(a.f if c else a.g)
assert h() == "s"
if c:
    assert r == "s"
else:
    assert r == 1
