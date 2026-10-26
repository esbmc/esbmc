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

    def func(self):
        b = B(self.c)
        return b.func()


for i in range(2):
    a = A()
    assert a.func() == 1
