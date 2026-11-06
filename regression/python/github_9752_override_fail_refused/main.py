class B:
    def g(self):
        return "string"


class E:
    def g(self):
        return "other"


class C:
    def f(self):
        return B()

    def h(self):
        return self.f().g()


class D(C):
    def f(self):
        return E()


xs = [C(), D()]
r = []
for x in xs:
    r.append(x.h())
assert r == ["string", "string"]
