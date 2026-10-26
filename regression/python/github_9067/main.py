class N:
    def __init__(self, b):
        self.b = b


def f(b):
    n = N(b)
    assert isinstance(n.b, int)


f(1)
