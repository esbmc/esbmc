class A:
    def __call__(self, z):
        return z + [1.5]


class B:
    def __call__(self, z):
        return z + [1]


a = A()
y = a([])
assert isinstance(y, list)
assert y[0] == 1.5

b = B()
w = b([4, 5])
assert len(w) == 3
assert w[2] == 1
