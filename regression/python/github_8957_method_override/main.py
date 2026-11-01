# A subclass overrides g; the instance is a B, so g is B.g and f is A.f.
class A:
    def f(self):
        return "a"

    def g(self):
        return 42


class B(A):
    def g(self):
        return 24

    def pick(self, c):
        if c:
            return self.f
        return self.g


b = B()
c = nondet_bool()
r = b.pick(c)()
if c:
    assert r == "a"
else:
    assert r == 24
