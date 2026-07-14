# A method that is not overridden in the instance's class or its direct
# parent must still resolve when inherited from a grandparent (MRO walk).
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
assert c.g() == 9   # g inherited from grandparent A: 10 - 1
assert c.f() == 20  # f overridden in C: 10 * 2
