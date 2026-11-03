class M:

    def __init__(self, b):
        self.b = b


class N:

    def __init__(self, b):
        self.m = M(b)


n = N(1)
assert type(n.m.b) is not int
