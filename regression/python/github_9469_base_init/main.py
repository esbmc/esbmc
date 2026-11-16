class Z:
    def __init__(self):
        self.q = 1


class Y(Z):
    pass


class C:
    def __init__(self, a):
        self.a = a


c = C(Y())
assert c.a.q == 1
