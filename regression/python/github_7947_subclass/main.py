# An inherited classmethod calling cls(...) constructs the subclass (#7947).
class Model:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, x):
        return cls(x)


class Scaled(Model):
    def __init__(self, x):
        self.x = x * 2


class Twice(Scaled):
    pass


m = Model.make(5)
s = Scaled.make(5)
t = Twice.make(5)
assert m.x == 5
assert s.x == 10
assert t.x == 10
assert isinstance(t, Twice)
