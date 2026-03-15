class C:
    def f(self):
        return 1

class B:
    def __init__(self):
        self.c = C()

class A:
    def __init__(self):
        self.b = B()

a = A()
x = a.b.c.f()
assert x == 1
