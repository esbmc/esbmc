# A classmethod that rebinds cls is rejected rather than verified (#7947).
class Other:
    def __init__(self, x):
        self.x = x + 1


class Model:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, x):
        cls = Other
        return cls(x)


m = Model.make(5)
assert m.x == 5
