# GitHub #8978: B's parameter only receives a.c, an unannotated self.c = C().
class C:
    def func(self):
        return 1


class B:
    def __init__(self, c):
        self.c = c

    def func(self):
        return self.c.func()


class A:
    def __init__(self):
        self.c = C()


a = A()
b = B(a.c)
assert b.func() == 1
x = b.c
assert x.func() == 1
