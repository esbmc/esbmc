class Vec:
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        return Vec(self.x + other.x)

    def __eq__(self, other):
        return self.x == other.x

    def __len__(self):
        return self.x

    def __getitem__(self, i):
        return self.x * i


a = Vec(2) + Vec(3)
assert a == Vec(5)
assert len(a) == 5
assert a[2] == 10
assert a
assert not Vec(0)
