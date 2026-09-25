# Same-named factory methods in different classes each return their own
# local variable, and an overriding classmethod constructs through cls (#7947).
class Counter:
    @staticmethod
    def make(x):
        obj = x + 1
        return obj


class Model:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, x):
        return cls(x)


class Shifted(Model):
    @classmethod
    def make(cls, x):
        obj = cls(x + 1)
        return obj


class Child(Shifted):
    pass


c = Counter.make(5)
s = Shifted.make(5)
k = Child.make(5)
assert c == 6
assert s.x == 6
assert k.x == 6
