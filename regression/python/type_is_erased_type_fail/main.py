class A:
    def __init__(self, v):
        self.a = v


class B(A):
    def get(self):
        return self.a


c = B(True).get()
assert type(c) is int
