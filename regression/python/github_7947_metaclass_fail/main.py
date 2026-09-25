# A classmethod inherited by a class with a metaclass is rejected rather than verified (#7947).
class Meta(type):
    def __call__(cls, x):
        return 42


class Model:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, x):
        return cls(x)


class Sub(Model, metaclass=Meta):
    def __init__(self, x):
        self.x = x * 2


s = Sub.make(5)
assert s.x == 10
