# Negative variant: the grandparent method resolves correctly, so an
# assertion on the wrong value must fail (guards against silently binding
# to a different/wrong method during the MRO walk).
class A:
    x = 10

    def f(self):
        return self.x + 1

    def g(self):
        return self.x - 1


class B(A):
    def f(self):
        return self.x + 2


class C(B):
    def f(self):
        return self.x * 2


c = C()
assert c.g() == 11  # wrong: g inherited from A returns 10 - 1 == 9
