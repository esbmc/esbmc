class A:
    def __init__(self, v):
        self.a = v


class B(A):
    def get(self):
        return self.a


c = B("hi").get()
assert type(c) == str
assert type(c) != int
d = B(2.5).get()
assert float is type(d)
