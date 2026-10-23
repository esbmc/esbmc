class B:
    def __call__(self, z):
        return z + [1]


b = B()
w = b([4, 5])
assert len(w) == 3
assert w[2] == 1
assert w[0] == 5
