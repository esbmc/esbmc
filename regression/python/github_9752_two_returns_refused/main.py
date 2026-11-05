class B:
    def g(self):
        return "string"


class E(B):
    pass


class C:
    def f(self, c):
        if c:
            return B()
        return E()

    def h(self):
        return self.f(True).g()


assert C().h() == "string"
