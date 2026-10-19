class A:
    def __init__(self, v):
        self.a = v


class B(A):
    pass


b = B(1)
assert isinstance(b.a, str)
