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


def run(o: C):
    return o.h()


assert run(D()) == "other"
assert run(C()) == "string"
