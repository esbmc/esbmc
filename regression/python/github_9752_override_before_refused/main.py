class B:
    def g(self):
        return "string"


class C:
    def f(self):
        return B()

    def h(self):
        return self.f().g()


class D(C):
    def f(self):
        return B()


assert D().h() == "string"
