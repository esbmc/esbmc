class Vec:
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        return Vec(self.x + other.x)

    def __eq__(self, other):
        return self.x == other.x


a = Vec(2) + Vec(3)
assert a == Vec(6)
