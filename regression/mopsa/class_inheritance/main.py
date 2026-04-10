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

a = A()
b = B()
c = C()
res = c.g()
