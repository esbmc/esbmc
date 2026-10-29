# As github_8957_method_override, but expects the overridden method to return 42.
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
if not c:
    assert r == 42
