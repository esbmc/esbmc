class B:
    def set(self):
        self.h = self.f

    def f(self):
        return 1


b = B()
b.set()
r = b.h()
assert r == 1
assert b.h() == 1
