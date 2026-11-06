class A:
    def f(self):
        return B()

    def g(self):
        return self.f().g()


class B(A):
    def g(self):
        return "string"


assert A().g() == "string"
