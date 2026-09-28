class V:
    def __init__(self, a):
        self.a = a

    def __add__(self, o):
        return V(self.a + o.a)


r = V(2) + V(3)
assert r.a == 6
