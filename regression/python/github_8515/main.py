class V:
    def __init__(self, x: int):
        self.x = x

    def __mul__(self, k):
        return V(self.x * k)

    def __rmul__(self, k):
        return V(self.x * k + 1)

    def __add__(self, k):
        return V(self.x + k)


v = V(2) * 3
assert v.x == 6
w = 4 * V(2)
assert w.x == 9
n = -5
u = V(2) + n
assert u.x == -3
