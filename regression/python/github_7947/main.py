# cls(...) inside a classmethod constructs the class it is called on (#7947).
class Model(object):
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, x):
        obj = cls(x)
        return obj


m = Model.make(5)
assert m.x == 5
