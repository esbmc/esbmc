class A:
    pass

class B:
    def __radd__(self, other):
        return 42

    def __rsub__(self, other):
        return 99

res1 = A() + B()
assert res1 == 42

res2 = A() - B()
assert res2 == 99