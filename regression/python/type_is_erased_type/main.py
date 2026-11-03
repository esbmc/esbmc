class A:
    def __init__(self, v):
        self.a = v


class B(A):
    def get(self):
        return self.a


c = B(1).get()
assert type(c) is int
assert type(c) is not bool
assert type(c) == int
