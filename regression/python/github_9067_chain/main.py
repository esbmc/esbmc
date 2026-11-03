class N:
    def __init__(self, b):
        self.b = b


def g(b):
    n = N(b)
    assert isinstance(n.b, int)
    assert not isinstance(n.b, str)


def f(b):
    g(b)


f(1)
