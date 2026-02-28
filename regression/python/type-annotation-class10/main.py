class A:

    def __init__(self, v):
        self.v = v


def f():
    x = A(10)
    return x


a = f()
assert a.v == 10
