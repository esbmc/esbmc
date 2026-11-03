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


class A2(A):
    pass


a = A2()
assert a.func() == 1
