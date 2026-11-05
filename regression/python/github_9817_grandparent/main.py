class C:
    def f(self, x: int) -> int:
        return x + 1

class D(C):
    pass

class E(D):
    pass

f = 0
e = E()
f = e.f(f)
assert f == 1
